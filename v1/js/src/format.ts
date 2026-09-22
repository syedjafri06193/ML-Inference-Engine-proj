/**
 * Just enough .lnn parsing to describe a model's inputs and outputs.
 *
 * The wasm module parses the file properly; this reads the same header and
 * tensor table so the JS API can expose names and shapes without a round
 * trip, and so `run({ image: ... })` can match names to indices.
 */

export const MAGIC = 0x314e4e4c;
export const HEADER_BYTES = 64;
export const TENSOR_BYTES = 40;

export const KIND_INPUT = 2;
export const KIND_OUTPUT = 3;

export interface TensorInfo {
  name: string;
  /** Element count, not bytes. */
  size: number;
  shape: number[];
  dtype: 'i8' | 'u8' | 'i32' | 'f32';
  scale: number;
  zeroPoint: number;
}

export interface ModelInfo {
  version: number;
  flags: number;
  nodes: number;
  tensors: number;
  arenaBytes: number;
  scratchBytes: number;
  inputs: TensorInfo[];
  outputs: TensorInfo[];
  /** Weights are 7-bit, so the relaxed kernel is safe to use. */
  sevenBit: boolean;
}

const DTYPES = ['i8', 'u8', 'i32', 'f32'] as const;

export function parseInfo(bytes: Uint8Array): ModelInfo {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < HEADER_BYTES || dv.getUint32(0, true) !== MAGIC) {
    throw new Error('not a .lnn model file');
  }
  const version = dv.getUint32(4, true);
  const flags = dv.getUint32(8, true);
  const nTensors = dv.getUint32(12, true);
  const nNodes = dv.getUint32(16, true);
  const arenaBytes = dv.getUint32(28, true);
  const scratchBytes = dv.getUint32(32, true);
  const tensorsOff = dv.getUint32(36, true);
  const weightsOff = dv.getUint32(44, true);
  const stringsOff = dv.getUint32(48, true);

  const inputs: TensorInfo[] = [];
  const outputs: TensorInfo[] = [];
  for (let i = 0; i < nTensors; i++) {
    const o = tensorsOff + i * TENSOR_BYTES;
    const kind = dv.getUint8(o + 2);
    if (kind !== KIND_INPUT && kind !== KIND_OUTPUT) continue;
    const dtype = DTYPES[dv.getUint8(o)] ?? 'i8';
    const rank = dv.getUint8(o + 1);
    const shape: number[] = [];
    for (let d = 0; d < rank; d++) shape.push(dv.getUint32(o + 4 + d * 4, true));
    const scaleOff = dv.getUint32(o + 24, true);
    const nameOff = dv.getUint32(o + 32, true);
    let end = stringsOff + nameOff;
    while (end < bytes.length && bytes[end] !== 0) end++;
    const info: TensorInfo = {
      name: new TextDecoder().decode(bytes.subarray(stringsOff + nameOff, end)),
      shape,
      size: shape.reduce((a, b) => a * b, 1),
      dtype,
      scale: dv.getFloat32(weightsOff + scaleOff, true),
      zeroPoint: dv.getInt32(o + 28, true),
    };
    (kind === KIND_INPUT ? inputs : outputs).push(info);
  }

  return {
    version,
    flags,
    nodes: nNodes,
    tensors: nTensors,
    arenaBytes,
    scratchBytes,
    inputs,
    outputs,
    sevenBit: (flags & 2) !== 0,
  };
}

/** Human-readable text for a runtime error code. */
export const ERRORS: Record<number, string> = {
  [-1]: 'the model file is truncated',
  [-2]: 'the model buffer is not 16-byte aligned',
  [-3]: 'not a .lnn file',
  [-4]: 'unsupported .lnn version',
  [-5]: 'malformed header',
  [-6]: 'malformed tensor descriptor',
  [-7]: 'malformed node',
  [-8]: 'the model uses an operator this build does not implement',
  [-9]: 'inconsistent shapes',
  [-10]: 'an attribute is out of range',
  [-11]: 'out of memory: the model does not fit in this browser',
  [-12]: 'no model is loaded',
  [-13]: 'the model declares too little scratch space',
  [-14]: 'a tensor has the wrong dtype',
};

export function errorMessage(code: number): string {
  return ERRORS[code] ?? `unknown error ${code}`;
}
