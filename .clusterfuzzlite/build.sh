#!/bin/bash -eu
# Build every harness under fuzz/ against THIS checkout.
#
# recusal has no runtime dependencies, so nothing is downloaded: the package is copied
# into the image's site-packages as it stands in the tree, and each harness is compiled
# with the image's own atheris. A harness is fuzz/fuzz_<name>.py; its seeds, if it has
# any, are fuzz/seeds/fuzz_<name>/.

site_packages="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
rm -rf "$site_packages/recusal"
cp -r "$SRC/recusal/recusal" "$site_packages/recusal"

for harness in "$SRC"/recusal/fuzz/fuzz_*.py; do
  name="$(basename "$harness" .py)"
  compile_python_fuzzer "$harness"
  if [ -d "$SRC/recusal/fuzz/seeds/$name" ]; then
    zip -j -q "$OUT/${name}_seed_corpus.zip" "$SRC/recusal/fuzz/seeds/$name"/*
  fi
done
