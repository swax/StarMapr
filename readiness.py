"""Infrastructure preflight only; never uses photos or performs face inference."""
import importlib.util
import os
import shutil
import socket
from pathlib import Path


def check_readiness():
    checks = {'host': socket.gethostname()}
    for tool in ('node', 'ffmpeg'):
        if not shutil.which(tool):
            raise RuntimeError(f'Preflight: {tool} is missing on {checks["host"]}')
        checks[tool] = 'available'
    if importlib.util.find_spec('yt_dlp') is None:
        raise RuntimeError('Preflight: yt-dlp is missing')
    checks['yt_dlp'] = 'available'
    if os.getenv('CELEBRITY_VERIFIER', 'off') == 'aws_required':
        spec = importlib.util.spec_from_file_location('aws_check', Path(__file__).with_name('91_check_aws_connection.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from celebrity_verifier import CelebrityVerifier
        client = CelebrityVerifier(Path('unused'), region=os.getenv('AWS_REGION', 'us-east-1'))._client()
        try:
            response = client.recognize_celebrities(Image={'Bytes': module.blank_png()})
            if response.get('ResponseMetadata', {}).get('HTTPStatusCode') != 200:
                raise RuntimeError('Unexpected AWS status')
        except Exception as exc:
            raise RuntimeError(f'Preflight: AWS unavailable ({type(exc).__name__}); refresh credentials or check permission') from None
        checks['aws'] = 'available (one blank-image request, separate from portrait budget)'
    else:
        checks['aws'] = 'disabled'
    return checks
