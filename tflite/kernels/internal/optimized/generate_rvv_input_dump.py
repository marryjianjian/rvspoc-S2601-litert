#!/usr/bin/env python3
"""Generates real-image input dumps for RVV model precision tests.

The dump format is intentionally simple and dependency-free:

  char[8]  magic: "RVVINP01"
  uint32   sample_count
  repeat sample_count:
    uint32 input_count
    repeat input_count:
      int32    TfLite type enum value
      uint32   rank
      int32[]  shape
      uint64   byte_size
      uint8[]  raw tensor bytes

Input images are uncompressed BMP files. The script uses `flatc` to read model
input metadata from the TFLite flatbuffer, then writes tensors that match the
model input type, shape, and quantization parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MAGIC = b"RVVINP01"

TFLITE_TYPE_TO_ENUM = {
    "FLOAT32": 1,
    "INT32": 2,
    "UINT8": 3,
    "INT64": 4,
    "STRING": 5,
    "BOOL": 6,
    "INT16": 7,
    "COMPLEX64": 8,
    "INT8": 9,
    "FLOAT16": 10,
    "FLOAT64": 11,
    "COMPLEX128": 12,
    "UINT64": 13,
    "RESOURCE": 14,
    "VARIANT": 15,
    "UINT32": 16,
    "UINT16": 17,
    "INT4": 18,
}


@dataclass(frozen=True)
class InputTensorSpec:
  type_name: str
  type_enum: int
  shape: list[int]
  scale: float | None
  zero_point: int | None
  min_value: float | None
  max_value: float | None


@dataclass(frozen=True)
class BmpImage:
  width: int
  height: int
  rgb: bytes


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Generate RVV precision input dumps from BMP images."
  )
  parser.add_argument("--model", required=True, help="Path to .tflite model.")
  parser.add_argument(
      "--schema",
      required=True,
      help="Path to TFLite schema.fbs used by flatc.",
  )
  parser.add_argument(
      "--images",
      nargs="+",
      required=True,
      help="BMP images. Each image becomes one sample.",
  )
  parser.add_argument("--output", required=True, help="Output dump path.")
  parser.add_argument(
      "--flatc",
      default="flatc",
      help="flatc executable. Default: flatc",
  )
  parser.add_argument(
      "--float_default_min",
      type=float,
      default=0.0,
      help="Fallback float input minimum when the model has no min metadata.",
  )
  parser.add_argument(
      "--float_default_max",
      type=float,
      default=1.0,
      help="Fallback float input maximum when the model has no max metadata.",
  )
  parser.add_argument(
      "--quant_default_min",
      type=float,
      default=-1.0,
      help="Fallback quantized real-value minimum when no min metadata exists.",
  )
  parser.add_argument(
      "--quant_default_max",
      type=float,
      default=1.0,
      help="Fallback quantized real-value maximum when no max metadata exists.",
  )
  return parser.parse_args()


def run_flatc(model_path: Path, schema_path: Path, flatc: str) -> dict:
  if shutil.which(flatc) is None:
    raise RuntimeError(f"flatc executable not found: {flatc}")
  with tempfile.TemporaryDirectory(prefix="rvv_input_dump_") as temp_dir:
    subprocess.run(
        [
            flatc,
            "--raw-binary",
            "--json",
            "--strict-json",
            "--defaults-json",
            "-o",
            temp_dir,
            str(schema_path),
            "--",
            str(model_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    json_path = Path(temp_dir) / (model_path.stem + ".json")
    with json_path.open("r", encoding="utf-8") as f:
      return json.load(f)


def first_value(values: object) -> object | None:
  if isinstance(values, list) and values:
    return values[0]
  return None


def read_input_specs(model_path: Path, schema_path: Path,
                     flatc: str) -> list[InputTensorSpec]:
  model = run_flatc(model_path, schema_path, flatc)
  subgraphs = model.get("subgraphs", [])
  if not subgraphs:
    raise RuntimeError("Model has no subgraph.")
  subgraph = subgraphs[0]
  tensors = subgraph.get("tensors", [])
  specs = []
  for input_index in subgraph.get("inputs", []):
    tensor = tensors[input_index]
    type_name = tensor.get("type")
    if type_name not in TFLITE_TYPE_TO_ENUM:
      raise RuntimeError(f"Unsupported input tensor type: {type_name}")
    quant = tensor.get("quantization", {}) or {}
    specs.append(
        InputTensorSpec(
            type_name=type_name,
            type_enum=TFLITE_TYPE_TO_ENUM[type_name],
            shape=[int(dim) for dim in tensor.get("shape", [])],
            scale=(
                float(first_value(quant.get("scale")))
                if first_value(quant.get("scale")) is not None else None
            ),
            zero_point=(
                int(first_value(quant.get("zero_point")))
                if first_value(quant.get("zero_point")) is not None else None
            ),
            min_value=(
                float(first_value(quant.get("min")))
                if first_value(quant.get("min")) is not None else None
            ),
            max_value=(
                float(first_value(quant.get("max")))
                if first_value(quant.get("max")) is not None else None
            ),
        )
    )
  return specs


def read_bmp(path: Path) -> BmpImage:
  data = path.read_bytes()
  if len(data) < 54 or data[:2] != b"BM":
    raise RuntimeError(f"{path} is not a BMP file.")
  pixel_offset = struct.unpack_from("<I", data, 10)[0]
  dib_header_size = struct.unpack_from("<I", data, 14)[0]
  if dib_header_size < 40:
    raise RuntimeError(f"{path}: unsupported BMP DIB header.")
  width = struct.unpack_from("<i", data, 18)[0]
  height_raw = struct.unpack_from("<i", data, 22)[0]
  planes = struct.unpack_from("<H", data, 26)[0]
  bits_per_pixel = struct.unpack_from("<H", data, 28)[0]
  compression = struct.unpack_from("<I", data, 30)[0]
  if width <= 0 or height_raw == 0:
    raise RuntimeError(f"{path}: invalid BMP dimensions.")
  if planes != 1 or bits_per_pixel not in (24, 32) or compression != 0:
    raise RuntimeError(
        f"{path}: only uncompressed 24/32-bit BMP is supported."
    )

  height = abs(height_raw)
  top_down = height_raw < 0
  bytes_per_pixel = bits_per_pixel // 8
  row_stride = ((width * bits_per_pixel + 31) // 32) * 4
  rgb = bytearray(width * height * 3)
  for y in range(height):
    src_y = y if top_down else (height - 1 - y)
    src = pixel_offset + src_y * row_stride
    for x in range(width):
      b = data[src + x * bytes_per_pixel]
      g = data[src + x * bytes_per_pixel + 1]
      r = data[src + x * bytes_per_pixel + 2]
      dst = (y * width + x) * 3
      rgb[dst:dst + 3] = bytes((r, g, b))
  return BmpImage(width=width, height=height, rgb=bytes(rgb))


def image_to_float_values(image: BmpImage, min_value: float,
                          max_value: float) -> Iterable[float]:
  scale = (max_value - min_value) / 255.0
  for pixel in image.rgb:
    yield min_value + float(pixel) * scale


def clamp(value: int, lo: int, hi: int) -> int:
  return max(lo, min(hi, value))


def tensor_bytes_for_image(image: BmpImage, spec: InputTensorSpec,
                           args: argparse.Namespace) -> bytes:
  if len(spec.shape) != 4 or spec.shape[0] != 1 or spec.shape[3] != 3:
    raise RuntimeError(
        f"Only NHWC image inputs with batch=1 and channels=3 are supported; "
        f"got shape {spec.shape}."
    )
  _, height, width, _ = spec.shape
  if image.width != width or image.height != height:
    raise RuntimeError(
        f"Image size {image.width}x{image.height} does not match model input "
        f"{width}x{height}."
    )

  if spec.type_name == "FLOAT32":
    min_value = (
        spec.min_value if spec.min_value is not None else args.float_default_min
    )
    max_value = (
        spec.max_value if spec.max_value is not None else args.float_default_max
    )
    return b"".join(
        struct.pack("<f", value)
        for value in image_to_float_values(image, min_value, max_value)
    )

  if spec.type_name in ("UINT8", "INT8"):
    if spec.scale is None or spec.zero_point is None or spec.scale == 0.0:
      raise RuntimeError(
          f"Quantized input {spec.type_name} requires scale and zero_point."
      )
    min_value = (
        spec.min_value if spec.min_value is not None else args.quant_default_min
    )
    max_value = (
        spec.max_value if spec.max_value is not None else args.quant_default_max
    )
    output = bytearray()
    for value in image_to_float_values(image, min_value, max_value):
      quantized = int(math.floor(value / spec.scale + spec.zero_point + 0.5))
      if spec.type_name == "UINT8":
        output.append(clamp(quantized, 0, 255))
      else:
        output.append(clamp(quantized, -128, 127) & 0xFF)
    return bytes(output)

  raise RuntimeError(f"Unsupported image input tensor type: {spec.type_name}")


def write_dump(path: Path, specs: list[InputTensorSpec],
               images: list[BmpImage], args: argparse.Namespace) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("wb") as f:
    f.write(MAGIC)
    f.write(struct.pack("<I", len(images)))
    for image in images:
      f.write(struct.pack("<I", len(specs)))
      for spec in specs:
        raw = tensor_bytes_for_image(image, spec, args)
        f.write(struct.pack("<iI", spec.type_enum, len(spec.shape)))
        for dim in spec.shape:
          f.write(struct.pack("<i", dim))
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)


def main() -> int:
  args = parse_args()
  model_path = Path(args.model)
  schema_path = Path(args.schema)
  image_paths = [Path(path) for path in args.images]
  output_path = Path(args.output)

  specs = read_input_specs(model_path, schema_path, args.flatc)
  images = [read_bmp(path) for path in image_paths]
  write_dump(output_path, specs, images, args)

  print(f"model={model_path}")
  print(f"output={output_path}")
  print(f"samples={len(images)}")
  for i, spec in enumerate(specs):
    print(
        f"input[{i}] type={spec.type_name} shape={spec.shape} "
        f"scale={spec.scale} zero_point={spec.zero_point} "
        f"min={spec.min_value} max={spec.max_value}"
    )
  print(f"bytes={output_path.stat().st_size}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
