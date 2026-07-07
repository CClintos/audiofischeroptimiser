import sys
import zlib
from pathlib import Path

XOR_KEY = b'ATFV6'


def _xor_repeat(data, key):
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def decode_bytes(path):
    raw = Path(path).read_bytes()
    unxored = _xor_repeat(raw, XOR_KEY)
    if len(unxored) < 4:
        raise ValueError('file too short to be a valid .pct6: %s' % path)
    declared = int.from_bytes(unxored[:4], 'big')
    xml_bytes = zlib.decompress(unxored[4:])
    if declared != len(xml_bytes):
        print('warning: declared length %d != decoded length %d -- verify this decode carefully'
              % (declared, len(xml_bytes)), file=sys.stderr)
    if not xml_bytes.lstrip().startswith(b'<ATF'):
        raise ValueError('decoded content does not look like tune XML -- this file may be '
                         'password-protected, or the key/container has changed on this '
                         'PC-Tool version. Do not trust this output.')
    return xml_bytes


def encode_bytes(xml_bytes, path):
    packed = len(xml_bytes).to_bytes(4, 'big') + zlib.compress(xml_bytes, 9)
    Path(path).write_bytes(_xor_repeat(packed, XOR_KEY))


def decode(path):
    return decode_bytes(path).decode('latin-1')


def encode(xml, path):
    encode_bytes(xml.encode('latin-1'), path)


def _selftest():
    sample = b'<ATF JPT="1" V="6.01.08"><OC><Fil T="17" F="110.00" Q="1.30" G="-4.00" FN="1"/></OC></ATF>'
    tmp = Path('_pct6_selftest.pct6')
    try:
        encode_bytes(sample, tmp)
        assert decode_bytes(tmp) == sample, 'decode_bytes/encode_bytes round-trip mismatch'

        sample_binary = sample[:-6] + bytes([0x93, 0xC1, 0xFE]) + sample[-6:]
        encode_bytes(sample_binary, tmp)
        text = decode(tmp)
        assert text.encode('latin-1') == sample_binary, 'latin-1 text view lost information'
        encode(text, tmp)
        assert decode_bytes(tmp) == sample_binary, 'decode()/encode() round-trip mismatch'

        print('SELFTEST PASSED (synthetic round-trip only -- also verify against a real file with '
              'decode_bytes(original) == decode_bytes(reencoded))')
    finally:
        tmp.unlink(missing_ok=True)


def _main():
    if len(sys.argv) < 2:
        print('usage: python pct6.py {decode <file.pct6> | encode <file.xml> <out.pct6> | selftest}')
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == 'selftest':
        _selftest()
    elif cmd == 'decode':
        xml_bytes = decode_bytes(sys.argv[2])
        out = Path(sys.argv[2]).with_suffix('.decoded.xml')
        out.write_bytes(xml_bytes)
        print('decoded ->', out)
        print(repr(xml_bytes[:200]))
    elif cmd == 'encode':
        xml_bytes = Path(sys.argv[2]).read_bytes()
        encode_bytes(xml_bytes, sys.argv[3])
        print('encoded ->', sys.argv[3])
    else:
        print('unknown command:', cmd)
        sys.exit(1)


if __name__ == '__main__':
    _main()
