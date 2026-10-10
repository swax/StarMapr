"""Carry source URLs through cache copies and into accepted model manifests."""

import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from validation import file_hash, write_json

MANIFEST = 'image-sources.json'


def domain(url):
    try:
        parsed = urlsplit(url or '')
        host = parsed.hostname
        if parsed.scheme not in ('http', 'https') or not host:
            return None
        host = host.lower().rstrip('.')
        return host.removeprefix('www.')
    except ValueError:
        return None


def read_sources(folder):
    path = Path(folder) / MANIFEST
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def save_search_sources(folder, results, query):
    """GoogleImagesSearch exposes the downloaded path, image URL and contextLink.

    Its downloader can overwrite identical basenames. Do not guess a source when
    multiple URLs share that path; the cache's actual bytes are still usable.
    """
    folder = Path(folder).resolve()
    sources = read_sources(folder)
    grouped = defaultdict(list)
    for result in results:
        if result.path:
            path = Path(result.path).resolve()
            if path.parent == folder and path.is_file():
                grouped[path].append(result)
    for path, matches in grouped.items():
        urls = {result.url for result in matches}
        if len(urls) != 1:
            sources.pop(path.name, None)
            continue
        result = matches[0]
        sources[path.name] = dict(sha256=file_hash(path), image_url=result.url,
                                  page_url=result.referrer_url, query=query, provider='google')
    write_json(folder / MANIFEST, sources)


def copied_source(path, sources):
    digest = file_hash(path)
    source = sources.get(Path(path).name, {})
    # Old cache/manual images have no known URL. Never infer one from filenames.
    if source.get('sha256') != digest:
        source = {}
    return dict(source, sha256=digest)


def accepted_sources(images):
    records = []
    manifests = {}
    for image in images:
        image = Path(image)
        if image.parent not in manifests:
            manifests[image.parent] = read_sources(image.parent)
        source = copied_source(image, manifests[image.parent])
        records.append(dict(source, filename=image.name,
                            page_domain=domain(source.get('page_url')),
                            image_domain=domain(source.get('image_url'))))
    return records
