"""vast.ai: rent the cheapest fitting GPU, serve VGGT-Omega on it, destroy it when done.

Talks to vast.ai's REST API directly (https://console.vast.ai, the endpoints the official CLI uses),
with the key from VAST_API_KEY or ~/.config/vastai/vast_api_key (`vastai set api-key`); the key is
never logged or sent to the host. Each rental gets a throwaway SSH key attached to that instance
only, kept under <state_dir>/keys/ and deleted on down(). The instance id is kept in
<state_dir>/gpu-vast.json, so a later up() reuses a running instance and a later down() (from any
process) destroys it; down() confirms from the instance listing that it is gone.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from camera_eval.gpu.base import Backend, Endpoint, RemoteHost, new_token, ssh_keygen, wait_healthy

API = "https://console.vast.ai"
KEY_FILE = Path.home() / ".config/vastai/vast_api_key"


class VastAPI:
    def __init__(self):
        self._key = os.environ.get("VAST_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else "")
        if not self._key:
            raise RuntimeError("no vast.ai key: set VAST_API_KEY or run `vastai set api-key KEY` (writes %s)" % KEY_FILE)

    def call(self, verb: str, path: str, query: dict | None = None, body: dict | None = None):
        url = API + path
        if query:
            url += "?" + "&".join("%s=%s" % (k, urllib.parse.quote_plus(v if isinstance(v, str) else json.dumps(v))) for k, v in query.items())
        data = json.dumps(body).encode() if body is not None else None
        last = None
        for attempt in range(6):
            req = urllib.request.Request(url, data=data, method=verb, headers={
                "Authorization": "Bearer " + self._key, "Content-Type": "application/json", "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    raw = r.read()
                return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as err:
                text = err.read()[:400].decode(errors="replace")
                if err.code == 429 or err.code >= 500:
                    last = "HTTP %d: %s" % (err.code, text)
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError("%s %s -> HTTP %d: %s" % (verb, path, err.code, text)) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
                last = str(err)
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("%s %s failed after retries: %s" % (verb, path, last))

    def user(self) -> dict:
        u = self.call("GET", "/api/v0/users/current")
        return {k: u.get(k) for k in ("credit", "balance", "email") if k in u}

    def instances(self) -> list[dict]:
        rows, query = [], {"select_filters": {}, "order_by": [{"col": "id", "dir": "asc"}], "limit": 25}
        while True:
            data = self.call("GET", "/api/v1/instances/", query)
            rows += data.get("instances") or []
            if not data.get("next_token"):
                return rows
            query["after_token"] = data["next_token"]

    def instance(self, iid: int) -> dict:
        return self.call("GET", "/api/v0/instances/%d/" % iid, {"owner": "me"}).get("instances") or {}

    def offers(self, vcfg: dict) -> list[dict]:
        q = {"verified": {"eq": True}, "external": {"eq": False}, "rentable": {"eq": True}, "rented": {"eq": False},
             "gpu_name": {"eq": vcfg["gpu_name"]}, "num_gpus": {"eq": 1}, "reliability": {"gt": 0.98},
             "cuda_max_good": {"gte": vcfg["min_cuda"]}, "inet_down": {"gte": 400}, "inet_up": {"gte": 200},
             "disk_space": {"gte": vcfg["disk_gb"]}, "direct_port_count": {"gte": 1},
             "order": [["dph_total", "asc"]], "type": "on-demand", "limit": 64, "allocated_storage": vcfg["disk_gb"]}
        offers = [o for o in self.call("POST", "/api/v0/bundles/", body=q).get("offers") or [] if o.get("dph_total", 99) <= vcfg["max_dph"]]
        near = [o for o in offers if str(o.get("geolocation", "")).endswith(("US", "CA"))]
        return near + [o for o in offers if o not in near]

    def create(self, offer_id: int, vcfg: dict, label: str) -> int:
        blob = {"client_id": "me", "image": vcfg["image"], "env": {}, "price": None, "disk": vcfg["disk_gb"], "label": label,
                "extra": None, "onstart": None, "image_login": None, "python_utf8": False, "lang_utf8": False,
                "use_jupyter_lab": False, "jupyter_dir": None, "force": False, "cancel_unavail": True,
                "template_hash_id": None, "user": None, "runtype": "ssh_direc ssh_proxy"}
        r = self.call("PUT", "/api/v0/asks/%d/" % offer_id, body=blob)
        if not r.get("success"):
            raise RuntimeError("create failed: %s" % r)
        return int(r["new_contract"])

    def attach_key(self, iid: int, pub: str) -> None:
        self.call("POST", "/api/v0/instances/%d/ssh/" % iid, body={"ssh_key": pub})

    def destroy(self, iid: int, log) -> bool:
        for _ in range(6):
            try:
                self.call("DELETE", "/api/v0/instances/%d/" % iid, body={})
            except RuntimeError as err:
                log("gpu_destroy_error", instance=iid, error=str(err)[:300])
            time.sleep(5)
            try:
                if not any(i.get("id") == iid for i in self.instances()):
                    return True
            except RuntimeError as err:
                log("gpu_list_error", error=str(err)[:300])
        return False


class VastBackend(Backend):
    kind = "vast"

    def __init__(self, cfg, state_dir, log, **overrides):
        super().__init__(cfg, state_dir, log)
        self.vcfg = {**cfg["gpu"]["vast"], **{k: v for k, v in overrides.items() if v is not None}}
        self.api = VastAPI()
        self.host: RemoteHost | None = None
        self.state = self.read_state()

    def _remote(self, state: dict) -> RemoteHost:
        keydir = self.state_dir / "keys" / state["label"]
        return RemoteHost(state["ssh"][0], state["ssh"][1], "root", keydir / "id_ed25519", keydir / "known_hosts",
                          self.gpu_cfg["remote_home"], self.log)

    def _rent(self) -> dict:
        label = "camera-eval-%s-%s" % (socket.gethostname().split(".")[0].lower(), datetime.now().strftime("%Y%m%d-%H%M%S"))
        pub = ssh_keygen(self.state_dir / "keys" / label / "id_ed25519", label)
        offers = self.api.offers(self.vcfg)
        if not offers:
            raise RuntimeError("no %s offer at or under $%.2f/h" % (self.vcfg["gpu_name"], self.vcfg["max_dph"]))
        for offer in offers[:8]:  # offers go fast: one seen a second ago may be rented already
            try:
                iid = self.api.create(offer["id"], self.vcfg, label)
            except RuntimeError as err:
                if "no_such_ask" in str(err) or "not available" in str(err):
                    self.log("gpu_offer_gone", offer=offer["id"])
                    continue
                raise
            state = {"instance": iid, "label": label, "dph": round(offer.get("dph_total", 0), 4), "gpu": offer.get("gpu_name"),
                     "geolocation": offer.get("geolocation"), "created": time.time(), "image": self.vcfg["image"]}
            self.write_state(state)  # from here on, down() can find and destroy it
            self.log("gpu_rented", instance=iid, gpu=state["gpu"], dph=state["dph"], geolocation=state["geolocation"])
            self.api.attach_key(iid, pub)
            return state
        raise RuntimeError("none of the first %d offers could be rented" % min(8, len(offers)))

    def _wait_running(self, iid: int, timeout: float) -> dict:
        t0, last = time.time(), None
        while time.time() - t0 < timeout:
            info = self.api.instance(iid)
            status = info.get("actual_status")
            if status != last:
                self.log("gpu_status", instance=iid, status=status)
                last = status
            if status == "running":
                return info
            if status in ("exited", "offline") or (info.get("status_msg") or "").lower().startswith("error"):
                raise RuntimeError("instance %d failed to start: %s %s" % (iid, status, (info.get("status_msg") or "")[:300]))
            time.sleep(10)
        raise TimeoutError("instance %d not running after %ds" % (iid, timeout))

    def up(self) -> Endpoint:
        if self.endpoint and self.host and self.host.tunnel_alive():
            return self.endpoint
        state = self.state
        if state.get("instance"):
            info = self.api.instance(state["instance"])
            if info.get("actual_status") != "running":
                self.log("gpu_stale_state", instance=state["instance"], status=info.get("actual_status"))
                self.down()
                state = {}
        fresh = not state.get("instance")
        try:
            if fresh:
                state = self._rent()
            info = self._wait_running(state["instance"], 1500)
            port22 = (info.get("ports") or {}).get("22/tcp")
            state["ssh"] = [info["public_ipaddr"], int(port22[0]["HostPort"])] if port22 else [info["ssh_host"], int(info["ssh_port"])]
            self.write_state(state)
            self.state = state
            self.host = self._remote(state)
            self.host.wait_ssh(600)
            self.host.setup(self.gpu_cfg)
            token = new_token()
            self.host.start_server(self.gpu_cfg, token)
            url = self.host.open_tunnel(self.gpu_cfg["remote_port"])
            try:
                health = wait_healthy(url, token, 600, self.log, "vast instance %d" % state["instance"])
            except TimeoutError:
                raise RuntimeError("server did not come up: %s" % self.host.server_log_tail()) from None
        except BaseException:
            if fresh:
                self.log("gpu_up_failed", instance=state.get("instance"))
                self.down()
            raise
        self._token = token
        self.endpoint = Endpoint(url, token, {"kind": "vast", "instance": state["instance"], "gpu": health.get("gpu"),
                                              "dph": state["dph"], "geolocation": state.get("geolocation")})
        return self.endpoint

    def ensure(self) -> None:
        if self.host and not self.host.tunnel_alive():
            self.log("gpu_tunnel_reopened", instance=self.state.get("instance"))
            self.endpoint.url = self.host.open_tunnel(self.gpu_cfg["remote_port"])

    def detach(self) -> None:
        if self.host:
            self.host.close_tunnel()
        self.log("gpu_kept", instance=self.state.get("instance"),
                 note="still billing: `camera-eval up` reuses it, `camera-eval down` destroys it")
        self.host, self.endpoint = None, None

    def cost_estimate_usd(self) -> float:
        if not self.state.get("created"):
            return 0.0
        return round((time.time() - self.state["created"]) / 3600 * self.state.get("dph", 0), 3)

    def down(self) -> None:
        state = self.read_state() or self.state
        if self.host:
            self.host.close_tunnel()
        iid = state.get("instance")
        if iid:
            cost = self.cost_estimate_usd() if self.state.get("created") else None
            if self.api.destroy(iid, self.log):
                self.log("gpu_destroyed", instance=iid, hours=round((time.time() - state.get("created", time.time())) / 3600, 3),
                         cost_usd_estimate=cost)
            else:
                self.log("gpu_destroy_unconfirmed", instance=iid,
                         action="destroy instance %s by hand at https://cloud.vast.ai/instances/" % iid)
                return  # keep the state so the next down() tries again
        if state.get("label"):
            shutil.rmtree(self.state_dir / "keys" / state["label"], ignore_errors=True)
        self.clear_state()
        self.state, self.host, self.endpoint = {}, None, None


def list_offers(cfg: dict, **overrides) -> list[dict]:
    vcfg = {**cfg["gpu"]["vast"], **{k: v for k, v in overrides.items() if v is not None}}
    return VastAPI().offers(vcfg)
