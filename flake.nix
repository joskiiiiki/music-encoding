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
      }
    );
}
