"""Exercise HTTP resume and integrity paths without depending on the network."""

import gzip
import io
import json

import pytest

from itdrec.data import download as module


class Response(io.BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Length": str(len(payload)), **(headers or {})}


@pytest.mark.parametrize("supports_resume", [True, False])
def test_resume_or_clean_restart_and_cached_checksum(tmp_path, monkeypatch, supports_resume):
    payload = gzip.compress(b"user_id,parent_asin,rating,timestamp\nu,i,5,123\n")
    offset = 12
    (tmp_path / "Pet_Supplies.csv.gz.part").write_bytes(payload[:offset])

    def urlopen(request, timeout):
        assert request.get_header("Range") == f"bytes={offset}-"
        if supports_resume:
            return Response(payload[offset:], 206,
                            {"Content-Range": f"bytes {offset}-{len(payload)-1}/{len(payload)}"})
        return Response(payload)

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    path = module.download("pet", tmp_path)
    assert path.read_bytes() == payload
    manifest = json.loads(path.with_suffix(".gz.json").read_text())
    assert manifest["bytes"] == len(payload)
    assert module.download("pet", tmp_path) == path  # Cached download must not open the network.
    path.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        module.download("pet", tmp_path)


def test_short_transfer_remains_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(module.urllib.request, "urlopen",
                        lambda *args, **kwargs: Response(b"short", headers={"Content-Length": "100"}))
    with pytest.raises(IOError, match="Incomplete download"):
        module.download("pet", tmp_path)
    assert not (tmp_path / "Pet_Supplies.csv.gz").exists()
    assert (tmp_path / "Pet_Supplies.csv.gz.part").exists()


def test_unexpected_range_does_not_append(tmp_path, monkeypatch):
    path = tmp_path / "Pet_Supplies.csv.gz.part"
    path.write_bytes(b"prefix")
    monkeypatch.setattr(module.urllib.request, "urlopen",
                        lambda *args, **kwargs: Response(b"tail", 206, {"Content-Range": "bytes 0-3/4"}))
    with pytest.raises(IOError, match="unexpected byte range"):
        module.download("pet", tmp_path)
    assert path.read_bytes() == b"prefix"
