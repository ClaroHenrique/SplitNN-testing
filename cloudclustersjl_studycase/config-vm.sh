apt-get install nano
curl -fsSL https://install.julialang.org | sh
source ~/.bashrc

mkdir teste
cd teste

git clone https://github.com/ClaroHenrique/SplitNN-testing.git
cd ~/teste/SplitNN-testing/cloudclustersjl_studycase

julia --project=. _main_single_node.jl




