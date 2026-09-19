#!/usr/bin/env python3
"""One AWS connectivity/permission check using a generated blank PNG, no photos."""

import argparse
import json
import os
from dotenv import load_dotenv
import struct
import zlib


def blank_png():
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    header = struct.pack('!2I5B', 128, 128, 8, 2, 0, 0, 0)
    pixels = (b'\0' + bytes([128, 128, 128]) * 128) * 128
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region', default=os.getenv('AWS_REGION', 'us-east-1'))
    args = parser.parse_args()
    import boto3
    from botocore.config import Config
    client = boto3.client('rekognition', region_name=args.region,
                          config=Config(connect_timeout=5, read_timeout=15,
                                        retries={'total_max_attempts': 1}))
    result = client.recognize_celebrities(Image={'Bytes': blank_png()})
    print(json.dumps(dict(connected=True, region=args.region,
                          http_status=result['ResponseMetadata']['HTTPStatusCode'],
                          recognized_faces=len(result.get('CelebrityFaces', [])),
                          unrecognized_faces=len(result.get('UnrecognizedFaces', [])),
                          input='generated blank gray PNG; no people or photos')))


if __name__ == '__main__':
    main()
