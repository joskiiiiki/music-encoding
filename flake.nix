{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            rocmSupport = true;
            allowUnfree = true;
          };
        };
        python = pkgs.python3.withPackages (ps: [
          ps.torch
          ps.torchvision
          ps.torchaudio
          ps.numpy
          ps.datasets          
          ps.transformers
          ps.ipython
          ps.matplotlib
          ps.pillow
          ps.chromadb  # vector DB for embedding lookup in the eval scripts
          ps.pyvis  # interactive (HTML) graph output for the network plots
          ps.gdown  # MTG-Jamendo download script dependency
        ]);
        # Lightweight python for the web API (`webapp/api`): the API reads exported
        # artifacts (sqlite + .npz), so it needs neither torch/ROCm nor chromadb.
        # Keeping them out makes this shell small and fast to evaluate, so it can be
        # entered alongside a training run without the ROCm closure.
        pythonWeb = pkgs.python3.withPackages (ps: [
          ps.numpy
          ps.fastapi
          ps.uvicorn
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [

            pkgs.mcp-server-memory
            pkgs.mcp-server-filesystem
            python
            pkgs.ty
            pkgs.ruff
            pkgs.runpodctl

          ];
        };

        # Web app: `nix develop .#web` (see webapp/README.md). Exports of the index
        # need the default shell instead, because they read chroma_db.
        devShells.web = pkgs.mkShell {
          packages = [
            pkgs.nodejs
            pkgs.pnpm
            pythonWeb
            pkgs.ruff
          ];
        };
      }
    );
}
