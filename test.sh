#!/bin/bash

cd "$(dirname "$0")"

if [[ "$1" == "-b" ]]; then
    bash build.sh
    if [[ $? -ne 0 ]]; then
        echo "Compilation failed during build.sh"
        exit 1
    fi
    ./build_out/custom_opp_euleros_aarch64.run
    shift
fi

cd ./case

if [[ "$1" == "-t" ]]; then
    shift
    
    for case_number in "$@"; do
        if [[ -d "case${case_number}" ]]; then
            cd "case${case_number}"
            echo -e "\e[32m进入目录 case${case_number} 并执行 run.sh\e[0m"
            chmod +x run.sh
            ./run.sh
            cd ..
        else
            echo -e "\e[31m目录 case${case_number} 不存在\e[0m"
        fi
    done
else
    for dir in */; do
        if [[ -d "$dir" ]]; then
            cd "$dir"
            if [[ -f "run.sh" ]]; then
                echo -e "\e[32m进入目录 $dir 并执行 run.sh\e[0m"
                chmod +x run.sh
                ./run.sh
            else
                echo -e "\e[31mrun.sh 文件在 $dir 中不存在\e[0m"
            fi
            cd ..
        fi
    done
fi

cd ..
