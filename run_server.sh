ifconfig
#export DEVICE="0"

while getopts w:t:d: flag
do
    case "${flag}" in
        u) username=${OPTARG};;
        a) age=${OPTARG};;
        f) fullname=${OPTARG};;
    esac
done
echo "Username: $username";
echo "Age: $age";
echo "Full Name: $fullname";
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:10230 -t 360 --reload detect_serv_m:app

