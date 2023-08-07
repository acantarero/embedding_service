# embedding_service

This repository provides the [instructor embedding model](https://instructor-embedding.github.io/) as an API service.  This can be run on CPU or GPU with any of the 3 model sizes, which you can set in the `.env` file.

## Installing

We have provided the script `debian_docker_install.sh` for installing docker on a Debian VM with support for GPU. This script was tested on Google Cloud with L4 GPU.

```
sh debian_docker_install.sh
```

To build an run the API service first copy the `.env` file

```
cp env-example.txt .env
```

and then launch with:

```
docker compose up
```

The API service documentation can be accessed at http://127.0.0.1:8000/docs.