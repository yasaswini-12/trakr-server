ifconfig
export DEVICE="cpu"
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:10230 -t 360 --reload detect_serv_m:app

