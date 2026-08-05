{
  description = "Ambiente de ML - Livro Aurélien Géron";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          (pkgs.python3.withPackages (python-pkgs: [
            python-pkgs.pandas
            python-pkgs.scikit-learn
            python-pkgs.tensorflow
            python-pkgs.matplotlib
            python-pkgs.jupyter
            python-pkgs.numpy
          ]))
        ];

        shellHook = ''
          echo "Ambiente pronto"
        '';
      };
    };
}
