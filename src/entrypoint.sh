#!/bin/sh

uv run gunicorn -b 0.0.0.0:80 app:server