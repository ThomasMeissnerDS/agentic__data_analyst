import base64


def _decode_plot_if_any(output_text: str, log: list, tmp_dir: str):
    """Detect base‑64 PNG/JPG → save file & log as TOOL_IMG."""
    text = output_text.strip()
    # PNG starts with iVBOR, JPG with /9j/4AA (both after base64)
    if text[:7] in {"iVBORw0", "/9j/4AA"}:
        img_path = f"{tmp_dir}/plot_{len([k for k,_ in log if k=='TOOL_IMG'])}.png"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(text))
        log.append(("TOOL_IMG", img_path))
        return True
    return False
