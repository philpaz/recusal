# The build image for ClusterFuzzLite: the fuzz harnesses under fuzz/, compiled against
# this checkout of recusal. Pinned by digest, as every other build input here is.
FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:21f6fab17ace4dc76e1c56de91ee95d3d03da021df94b41056ba036f6c6f55e9
COPY . $SRC/recusal
WORKDIR $SRC/recusal
COPY .clusterfuzzlite/build.sh $SRC/build.sh
