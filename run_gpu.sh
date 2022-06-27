ifconfig
#export DEVICE="0"

while getopts w:t:d: flag
do
    case "${flag}" in
        w) worker=${OPTARG};;
        t) timeout=${OPTARG};;
        d) device=${OPTARG};;
    esac
done
export DEVICE="$device"


gunicorn -k uvicorn.workers.UvicornWorker -w $worker -b 0.0.0.0:10230 -t $timeout --reload detect_serv_m:app

