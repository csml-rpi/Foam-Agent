"""
# Usage

Initialize your client as follows:

```python

import tracking_aws

bedrock_runtime = tracking_aws.new_default_client()
```

Your usage will be tracked in a local file called `usage_nrel_aws.json`.
"""
from __future__ import annotations
from math import nan
import itertools
import os
import io
import json
import pathlib
from typing import Union, Dict
import pathlib
from typing import Any
from contextlib import contextmanager

import boto3

Usage = Dict[str, Union[int, float]]  # TODO: could have used Counter class
default_usage_file = pathlib.Path("usage_nrel_aws.json")

CLAUDE_3_5_HAIKU = 'arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/g47vfd2xvs5w'
CLAUDE_3_5_SONNET = 'arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/56i8iq1vib3e'

# prices match https://aws.amazon.com/bedrock/pricing/ as of January 2025.
# but they don't consider discounts for caching or batching.
# Not all models are listed in this file, nor is the fine-tuning API.
pricing = {  # $ per 1,000 tokens
    CLAUDE_3_5_HAIKU: {'input': 0.0008, 'output': 0.004},
    CLAUDE_3_5_SONNET: {'input': 0.003, 'output': 0.015},
}

# Default models.  These variables can be imported from this module.
# Even if the system that is being evaluated uses a cheap default_model.
# one might want to evaluate it carefully using a more expensive default_eval_model.
default_model = CLAUDE_3_5_HAIKU
default_eval_model = CLAUDE_3_5_HAIKU


# A context manager that lets you temporarily change the default models
# during a block of code.  You can write things like
#     with use_model('arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/g47vfd2xvs5w'):
#        ...
#
#     with use_model(eval_model='arn:aws:bedrock:us-west-2:991404956194:application-inference-profile/g47vfd2xvs5w'):
#        ...
@contextmanager
def use_model(model: str = default_model, eval_model: str = default_eval_model):
    global default_model, default_eval_model
    save_model, save_eval_model = default_model, default_eval_model
    default_model, default_eval_model = model, eval_model
    try:
        yield
    finally:
        default_model, default_eval_model = save_model, save_eval_model


def track_usage(client: boto3.client, path: pathlib.Path = default_usage_file) -> boto3.client:
    """
    This method modifies (and returns) `client` so that its API calls
    will log token counts to `path`. If the file does not exist it
    will be created after the first API call. If the file exists the new
    counts will be added to it.

    The `read_usage()` function gets a Usage object from the file, e.g.:
    {
        "cost": 0.0022136,
        "input_tokens": 16,
        "output_tokens": 272
    }

    >>> client = boto3.client('bedrock')
    >>> track_usage(client, "example_usage_file.json")
    >>> type(client)
    <class 'botocore.client.BaseClient'>

    """
    old_invoke_model = client.invoke_model

    def tracked_invoke_model(*args, **kwargs) -> Any:
        response = old_invoke_model(*args, **kwargs)
        old: Usage = read_usage(path)
        new, response_body = get_usage(response, model=kwargs.get('modelId', None))
        _write_usage(_merge_usage(old, new), path)
        return response_body

    client.invoke_model = tracked_invoke_model  # type:ignore
    return client


def get_usage(response, model=None) -> Usage:
    """Extract usage info from an AWS Bedrock response."""
    response_body = json.loads(response['body'].read().decode())
    usage: Usage = {'input_tokens': response_body['usage']['input_tokens'],
                    'output_tokens': response_body['usage']['output_tokens']}

    # add a cost field
    try:
        costs = pricing[model]  # model name passed in request (may be alias)
    except KeyError:
        raise ValueError(f"Don't know prices for model {model} or {response.model}")

    cost = (usage.get('input_tokens', 0) * costs['input']
            + usage.get('output_tokens', 0) * costs['output']) / 1_000
    usage['cost'] = cost
    return usage, response_body


def read_usage(path: pathlib.Path = default_usage_file) -> Usage:
    """Retrieve total usage logged in a file."""
    if os.path.exists(path):
        with open(path, "rt") as f:
            return json.load(f)
    else:
        return {}


def _write_usage(u: Usage, path: pathlib.Path):
    with open(path, "wt") as f:
        json.dump(u, f, indent=4)


def _merge_usage(u1: Usage, u2: Usage) -> Usage:
    return {k: u1.get(k, 0) + u2.get(k, 0) for k in itertools.chain(u1, u2)}


def new_default_client(default='boto3') -> boto3.client:
    """Set the `default_client` to a new tracked client, based on the current
    aws credentials. If your credentials change you should call this method again."""
    global default_client
    default_client = track_usage(
        boto3.client('bedrock-runtime', region_name='us-west-2'))  # create a client with default args, and modify it
    # so that it will store its usage in a local file
    return default_client


# new_default_client()       # set `default_client` right away when importing this module

if __name__ == "__main__":
    import doctest

    doctest.testmod()