# Track 3 submission image: one native binary `simulate` (also `simulate-batch`) on PATH, on an empty base (a static binary
# needs nothing else: a single small layer to create and remove).
FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ zlib1g-dev && rm -rf /var/lib/apt/lists/*
COPY sim/ /src/
COPY train/ /train/
ARG CXXFLAGS="-O2 -std=c++17 -march=x86-64-v2 -mtune=generic -ffp-contract=off -fno-plt -pthread -DT3_SEALED -fno-ident -g0"
# profile-guided build: instrumented binary runs a few public scenarios, then the final build uses the profile
# (both builds use the same output name so the .gcda profile is found; -Werror=missing-profile guards it)
RUN mkdir -p /pgo && g++ $CXXFLAGS -DT3_PGO_GEN -fprofile-generate -fprofile-dir=/pgo -fprofile-update=atomic -o /simulate /src/main.cpp -lz \
 && for f in /train/*.json; do /simulate simulate --config $f --out /tmp/o/trace.parquet >/dev/null; done \
 && T3_HELPER=0 /simulate simulate --config /train/t3-as01-base-mix.json --out /tmp/o/trace.parquet >/dev/null \
 && g++ $CXXFLAGS -fprofile-use -fprofile-dir=/pgo -fprofile-correction -Werror=missing-profile -static -o /simulate /src/main.cpp -lz \
 && strip -s -R .comment -R .note.gnu.build-id -R .note.ABI-tag /simulate \
 && mkdir -p /root-fs/usr/local/bin /root-fs/tmp && chmod 1777 /root-fs/tmp \
 && cp /simulate /root-fs/usr/local/bin/simulate && ln -s simulate /root-fs/usr/local/bin/simulate-batch

FROM scratch
LABEL qfbench2.interface_version="2.0"
ENV PATH=/usr/local/bin
COPY --from=build /root-fs/ /
WORKDIR /tmp
CMD ["simulate", "--help"]
