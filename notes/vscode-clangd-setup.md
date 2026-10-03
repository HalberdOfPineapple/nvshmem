# VSCode C++/CUDA setup (clangd + clang-tidy)

Set up 2026-09-28. Gives go-to-definition, hover, completion, compiler errors as you type, and lint warnings for `.cu`/`.cuh`/`.cpp`/`.h` in this repo, without a GPU or CUDA on the host.

## How it fits together

- **clangd** (VSCode extension `llvm-vs-code-extensions.vscode-clangd`) is the language server. It runs the real clang front end on each file, so errors match what a compiler would say, and it understands CUDA (`__global__`, `<<<...>>>`, `__CUDA_ARCH__`).
  - The Microsoft C/C++ extension (cpptools) stays installed for debugging, but its IntelliSense is turned off because it handles CUDA poorly and would show duplicate squiggles.
- **clang-tidy** is the linter. clangd runs it automatically, and the rules live in `.clang-tidy`: bug-finding and performance checks, not style rewrites.
- **Compile database** (`compile_commands.json`): the exact compiler command CMake uses for each source file. clangd needs it to know the include paths and macros. CMake writes it to `build/compile_commands.json` because the build is configured with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`.
- **Why a translated copy?** The build runs inside the `nvcr.io/nvidia/pytorch:25.06-py3` container with nvcc, so CMake's database:
  1. hides include paths in nvcc `--options-file *.rsp` files,
  2. uses nvcc-only flags (`--generate-code`, `-rdc=true`, `-Xcompiler`),
  3. points at container-only paths (`/usr/local/cuda`, `/opt/hpcx/ompi`).

  `.vscode/clangd_compdb.py` fixes all three and writes `build/clangd/compile_commands.json`. `.clangd` tells clangd to use that copy.
- **Host copy of headers:** `~/sysroots/pytorch-25.06/` holds the container's CUDA 12.9 headers and libdevice, plus the HPC-X MPI headers (37 MB). The container itself lives on node-local disk and disappears when the job ends.
- **Headers:** headers are not in CMake's database, so clangd normally borrows flags from a "nearby" source file. That is often a host-only `.cpp`, which parses the header as plain C++ with `__CUDA_ARCH__` undefined. The script therefore adds explicit entries for every `.cuh` (and `.h`/`.hpp` under a `device/` dir), copying flags from the closest `.cu` file. For headers under `src/include` it also pre-includes `nvshmem_host.h` and `device/nvshmem_device_macros.h`, because many NVSHMEM headers are not self-contained. 39 of 46 such headers parse cleanly. The rest still show errors when opened alone:
  - `logical_endpoint_device.cuh`, `nvshmemi_region_device.cuh`, `nvshmemi_path_predicates.cuh`, `nvshmemi_h_to_d_rma_defs.cuh` and `nvshmemi_common_device_defines.cuh` need declarations from other headers.
  - `gdaki_device.cuh` and `ibgda_device.cuh` belong to transports that are disabled in this build (no DOCA headers; GNU `typeof`).
- CUDA files are parsed as **device code for `sm_100`** (GB200), so `#if __CUDA_ARCH__ >= ...` kernel branches are live. To change this, edit `GPU_ARCH` in the script.

## Files

| File | Purpose |
|---|---|
| `.vscode/settings.json` | clangd path/args, cpptools IntelliSense off, CMake Tools auto-configure off |
| `.vscode/tasks.json` | tasks "clangd: refresh compile DB" and "clang-tidy: current file" |
| `.vscode/clangd_compdb.py` | nvcc → clang translation of the compile database |
| `.clangd` | clangd config (which DB to use, suppressed diagnostics, inlay hints) |
| `.clang-tidy` | lint rules |
| `~/envs/buildtools/bin/{clangd,clang-tidy,clang-format}` | LLVM 22 tools from pip wheels (the extension's auto-download has no aarch64 build) |

All of these are listed in `.git/info/exclude`, so they never appear in `git status`.

## When you reconfigure CMake

After any `cmake` run on `build/` (inside the container), regenerate the translated DB:

```bash
python3 .vscode/clangd_compdb.py        # or: Ctrl+Shift+P → "Tasks: Run Task" → "clangd: refresh compile DB"
```

Then run Ctrl+Shift+P → "clangd: Restart language server".

For a fresh build dir, keep compile-command export on when configuring (inside the container):

```bash
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ...
```

## Known nvcc-vs-clang differences (suppressed in `.clangd`)

NVSHMEM's own headers contain two patterns that nvcc accepts but clang rejects as errors, so they are hidden:

- `cuda_ovl_target`: in the device pass, `host/nvshmem_macros.h` makes host-API declarations `__host__ __device__`, and `device/nvshmem_defines.h` then redefines them as `__device__`.
- `static_non_static`: `host/nvshmemx_coll_api.h` declares functions non-static, and `device/nvshmemx_coll_defines.cuh` redefines them with `NVSHMEMI_STATIC`.

The CLI `clang-tidy` can't suppress them, so the task filters those lines out.

## Not configured

- **Formatting:** the repo has no `.clang-format`, so format-on-save is off, to avoid reformatting upstream code. `clang-format` is installed if you want to add a style file later.
- **CUDA debugging** (Nsight extension / cuda-gdb) would need to run inside the container.
