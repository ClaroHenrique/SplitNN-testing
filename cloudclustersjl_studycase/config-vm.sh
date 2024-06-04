touch ~/.no_auto_tmux
apt-get install nano
curl -fsSL https://install.julialang.org | sh
source ~/.bashrc

mkdir teste
cd teste

git clone https://github.com/ClaroHenrique/SplitNN-testing.git
cd ~/teste/SplitNN-testing/cloudclustersjl_studycase
git checkout gpu-support

julia --project=. _main_single_node.jl




