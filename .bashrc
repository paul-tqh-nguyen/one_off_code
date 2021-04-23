
# Convenience & Workflow 

export PS1="\u@\h:\`pwd\`$ "
alias lt="ls -ltr"
alias yes="yes | head -n 1000"
alias untar="tar -xvf"

alias update-settings="pushd ~/code/one_off_code/ ; git pull; git add python_startup.py .emacs .bashrc ; git commit -m \"Update .bashrc and .emacs and python_startup.py files. Timestamp: $(date +\"%T\")\" ; git push ; source ~/code/one_off_code/.bashrc  ; popd"
alias store-git-credentials="git config --global credential.helper store"

function hgrep {
    history | grep -i $@ | awk '{ $1="" ; print}' | sort | uniq
}

function filesize {
    num_bytes=$(cat $1 | wc -c)
    if [ "1024" -gt "$num_bytes" ]
    then
	echo $(echo "$num_bytes") bytes
    else
	echo $(echo $num_bytes | numfmt --to=iec-i)
    fi
}  

function dirsize {
    echo $(du -h $1 | tail -n 1 | awk '{print $1}')
}

function set-title() {
  if [[ -z "$ORIG" ]]; then
    ORIG=$PS1
  fi
  TITLE="\[\e]2;$*\a\]"
  PS1=${ORIG}${TITLE}
}

# Spell Checking

function files-with-suffix {
    find . -type f -name "*.$1"
}

function spellcheck {
    for file in $@
    do
	printf "\n\n"
	echo $file
	cat $file | aspell -a | grep -v "^\*" | grep -v "^$" | grep "&" | cut -d" " -f2 \
		| grep -v "^br$" \
		| grep -v "^href$" \
		| grep -v "^hspace$" \
		| grep -v "^img$" \
		| grep -v "^png$" \
		| grep -v "^src$" \
		| grep -v "^th$" \
		| grep -v "^vw$" \
		| grep -v "^$" \
		| grep -v "^$" \
		| grep -v "^$" \
		| grep -v "^$" \
		| grep -v "^$" \
		| grep -v "^$" \
		| grep -v "^$" \
		| grep -v "^$" \
		| grep -v "^$"
	printf "\n\n"
    done
}

# Development

function git-black {
    black $(git status | grep modified | grep "\.py$" | cut -d":" -f2)
}

function git-add-mod {
    git add -f $(git status | grep modified | cut -d":" -f2)
}
alias gam="git-add-mod"
alias update-via-upstream="git pull --rebase upstream main && git fetch upstream && git push"

alias update-notebooks="pushd /home/pnguyen/code/hive-project/2020-Q2 ; for e in \$(ls */ | grep \":\" | cut -d\"/\" -f1) ; do tar -czvf \$e.tar.gz \$e ; done ; popd ; pushd /home/pnguyen/code/metagraph/docs/_downloads/notebooks ; cp /home/pnguyen/code/hive-project/2020-Q2/*.tar.gz . ; popd"

alias ipynb-to-py="jupyter nbconvert --to script"
export PYTHONSTARTUP=$HOME/code/one_off_code/python_startup.py

# OS Specific Basic Needs

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    bind 'set bell-style none'
    alias open="xdg-open"
    export PATH=/usr/local/cuda/bin${PATH:+:$PATH}
    export PATH=/home/pnguyen/.local/bin${PATH:+:$PATH}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export PATH=$PATH:~/scripts/:~/bin/
elif [[ "$OSTYPE" == "darwin"* ]]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    export PATH=/usr/local/bin${PATH:+:$PATH}
    export PATH=/Users/pnguyen/anaconda3/condabin${PATH:+:$PATH}
    export PATH=/opt/anaconda3/bin${PATH:+:$PATH}
    alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'
    alias egrep='egrep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias grep='grep --color=auto'
    alias l='ls -CF'
    alias la='ls -A'
    alias ll='ls -alF'
    alias lt='ls -ltr'
else
    echo "Could not detect OS flavor."
fi

# Anaconda Utilities

alias ssh-cuda="ssh pnguyen@colo-dgx-01.corp.continuum.io"
alias ssh-tunnel="ssh pnguyen@colo-dgx-01.corp.continuum.io -NL 8080:localhost:8080"

alias pytest="pytest --cov-report=html -rs"

alias init-impact="source ~/.bashrc ; cd ~/code/impact_of_attention && conda env create ; conda activate impact"
alias del-impact="for i in \$(seq 1 5); do conda deactivate ; done ; ~/code/impact_of_attention && conda env remove --name impact"
alias fresh-impact="del-impact && init-impact"

alias init-reuters="source ~/.bashrc ; cd ~/code/reuters_topic_labelling && conda env create ; conda activate reuters"
alias del-reuters="for i in \$(seq 1 5); do conda deactivate ; done ; ~/code/reuters_topic_labelling && conda env remove --name reuters"
alias fresh-reuters="del-reuters && init-reuters"

alias install-mg-libraries-not-yet-confirmed-to-be-included-in-environment-yml=": \

&& :"
alias goto-mg="cd ~/code/metagraph/"
alias init-mg="source ~/.bashrc ; for i in \$(seq 1 5); do conda deactivate ; done ; goto-mg && conda env create ; conda activate mg \
&& yes | conda clean --packages --tarballs \
&& yes | conda install -c conda-forge grblas pandas jupyter matplotlib \
&& pre-commit install \
&& python setup.py develop \
&& :"
alias del-mg="for i in \$(seq 1 5); do conda deactivate ; done ; goto-mg && conda env remove --name mg"
alias fresh-mg="del-mg && init-mg"

alias goto-mgc="cd ~/code/metagraph-cuda/"
alias init-mgc="source ~/.bashrc ; for i in \$(seq 1 5); do conda deactivate ; done ; goto-mgc && conda env create ; conda activate mgc \
&& yes | conda clean --packages --tarballs \
&& yes | conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph cudatoolkit=10.1 \
&& yes | conda install numba==0.48.0 pandas==0.25.3 networkx numpy scipy conda-forge::grblas jupyter matplotlib \
&& pre-commit install \
&& python setup.py develop \
&& :"
alias del-mgc="for i in \$(seq 1 5); do conda deactivate ; done ; goto-mgc && conda env remove --name mgc"
alias fresh-mgc="del-mgc && init-mgc"

if [ $(hostname) = "demouser-DGX-Station" ] ; then
    export PATH=/home/pnguyen/miniconda3/bin${PATH:+:$PATH}
fi
