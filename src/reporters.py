import fs
from uv.fs.reporter import FSReporter
from typing import Optional, Dict, List
import uv.types as t
import uv.util as u
import uv.reporter as r
import torch

def gcloud_reporter(prefix: str, job_name: str):
  """Returns a reporter which persists metrics in GCloud in jsonl format.
  """

  if prefix.endswith("/"):
    prefix = prefix[:-1]

  cloud_path = f"{prefix}/{job_name}?strict=False"
  gcsfs = fs.open_fs(cloud_path)
  return FSReporter(gcsfs).stepped()

def local_reporter(folder: str, job_name: str):
  """Returns a reporter implementation that persists metrics on the local
  filesystem in jsonl format.

  """

  local_path = fs.path.join(folder, job_name)
  return FSReporter(local_path).stepped()

def build_reporters(
    save_config: Dict,
    data_store: Optional[Dict[str, List[t.Metric]]] = None,
    logging: bool = True) -> r.AbstractReporter:

  """Returns a dict of namespace to reporter."""

  base = r.MemoryReporter(data_store).stepped()
  log = r.LoggingReporter()
  base = base.plus(log)

  base_loc = save_config['save_location']
  job_name = save_config['job_name']

  if is_gcloud(base_loc):
    gcloud = gcloud_reporter(base_loc, job_name)
    base = base.plus(gcloud)
  else:
    local = local_reporter(base_loc, job_name)
    base = base.plus(local)

  base = base.map_values(lambda step, v: u.to_metric(v))

  return base

def prefix_reporters(
    reporter: r.AbstractReporter,
    prefixes: List[str]) -> Dict[str, r.AbstractReporter]:

  """Prefix reporters add the specified prefix to each metric reported"""

  return {prefix: reporter.with_prefix(prefix) for prefix in prefixes}

"""
Code below this line ismeant to be replaced, and is a placeholder currently
necessary because the UV library does not have the ability to write arbitrary
files to a particular location
"""

import json
import os
from google.cloud import storage

def save_config(
    config: Dict) -> None:

    base_loc = config['save_config']['save_location']
    job_name = config['save_config']['job_name']
    folder = fs.path.join(base_loc, job_name)

    file_name = 'config.json'
    if is_gcloud(base_loc):
        bucket_name, bucket_address = extract_bucket_name(base_loc)

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)

        temp_filename = 'config_TEMPFILE.json'
        with open(temp_filename, 'w') as f:
            json.dump(config, f, indent=2)

        bucket_path = fs.path.join(bucket_address, job_name, file_name)
        bucket_path = bucket_path
        print(bucket_path)

        blob = bucket.blob(bucket_path)
        blob.upload_from_filename(temp_filename)

        os.remove(temp_filename)

    else:
        # local folder
        save_path = fs.path.join(folder, file_name)

        if not os.path.exists(folder):
          os.makedirs(folder)

        with open(save_path, 'w') as fout:
            json.dump(config, fout, indent=2)

def save_dict(
    config: Dict, to_save: Dict, file_name: str) -> None:
    """ saves a dictionary to the proper location, specified in config """

    base_loc = config['save_config']['save_location']
    job_name = config['save_config']['job_name']
    folder = fs.path.join(base_loc, job_name)

    extension = 'pth'

    if is_gcloud(base_loc):
        bucket_name, bucket_address = extract_bucket_name(base_loc)

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)

        temp_filename = f'{file_name}_TEMPFILE.{extension}'
        torch.save(to_save, temp_filename)

        bucket_path = fs.path.join(bucket_address, job_name, file_name)

        blob = bucket.blob(bucket_path)
        blob.upload_from_filename(temp_filename)

        os.remove(temp_filename)

    else:
        # local folder
        save_path = fs.path.join(folder, f'{file_name}.{extension}')

        if not os.path.exists(folder):
          os.makedirs(folder)

        torch.save(to_save, save_path)

def is_gcloud(folder: str) -> bool:

    gcloud_string = 'gs://'
    return gcloud_string in folder

def extract_bucket_name(folder: str) -> str:

    gcloud_string = 'gs://'
    assert folder[:len(gcloud_string)] == gcloud_string

    postfix = folder[len(gcloud_string):]
    if '/' not in postfix:
        return postfix, None
    else:
        index = postfix.index('/')
        return postfix[:index], postfix[index+1:]

def insert_in_str(source_str, insert_str, pos):
    return source_str[:pos] + insert_str + source_str[pos:]
