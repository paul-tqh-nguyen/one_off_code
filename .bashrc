
# Convenience & Workflow 

export PS1="\u@\h:\`pwd\`$ "
alias lt="ls -ltr"
alias yes="yes | head -n 1000"

alias update-settings="pushd ~/code/one_off_code/ ; git pull; git add .bashrc .emacs python_startup.py ; git commit -m \"Update .bashrc and .emacs and python_startup.py files.\" ; git push ; source ~/.bashrc ; popd"
alias store-git-credentials="git config --global credential.helper store"

function git-add-mod {
    git add $(git status | grep modified | cut -d":" -f2)
}

alias ipynb-to-py="jupyter nbconvert --to script"
export PYTHONSTARTUP=$HOME/code/one_off_code/python_startup.py

function filesize {
    num_bytes=$(cat $1 | wc --bytes)
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

alias ssh-cuda="ssh 192.168.131.229"

alias install-mg-libraries-not-yet-confirmed-to-be-included-in-environment-yml=": \
&& yes | conda install pandas \
&& :"
alias goto-mg="cd ~/code/metagraph/"
alias init-mg="goto-mg && conda env create ; conda activate mg && install-mg-libraries-not-yet-confirmed-to-be-included-in-environment-yml && pre-commit install && python setup.py develop"
alias del-mg="goto-mg && conda env remove --name mg"
alias fresh-mg="del-mg && init-mg"

alias install-mgc-libraries-not-yet-confirmed-to-be-included-in-environment-yml=": \
&& install-mg-libraries-not-yet-confirmed-to-be-included-in-environment-yml \
&& conda install ~/dump/metagraph-0.0.1-py3.7h39e3cac_g15c13c6_12.tar.bz2 \
&& yes | conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cugraph cudatoolkit=10.1 \
&& :"
alias goto-mgc="cd ~/code/metagraph-cuda/"
alias init-mgc="goto-mgc && conda env create ; conda activate mgc && install-mgc-libraries-not-yet-confirmed-to-be-included-in-environment-yml && pre-commit install && python setup.py develop"
alias del-mgc="goto-mgc && conda env remove --name mgc"
alias fresh-mgc="del-mgc && init-mgc"

if [ $(hostname) = "demouser-DGX-Station" ] ; then
    export PATH=/home/pnguyen/miniconda3/bin${PATH:+:$PATH}
fi
