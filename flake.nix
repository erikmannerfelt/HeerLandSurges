{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    nixrik = {
      url = "gitlab:erikmannerfelt/nixrik";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {self, nixpkgs, nixrik}:
    nixrik.lib.eachDefaultSystem (system: (
      let
        pkgs = nixpkgs.legacyPackages.${system};

        my-python = nixrik.packages.${system}.python_from_requirements {python_packages = pkgs.python310Packages;} ./requirements.txt;

      in {
        devShells.default = pkgs.mkShell {
          name = "HeerLandSurges";
          buildInputs = with pkgs; [
            my-python
          ];
        };
      }
    ));
}