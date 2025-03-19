import sys
import os
import shutil
import json
import logging
import subprocess
import shlex
import time
import requests


MODEL_DIR_BASE = os.getcwd()+"/models"
NGINX_DIR_BASE = os.getcwd()+"/nginx"
RESULTS_DIR_BASE = os.getcwd()+"/results/result_"+str(int(time.time()))
SERVED_MODEL_NAME = "model_in_test"
HUGGING_FACE_HUB_TOKEN = os.environ.get('HUGGING_FACE_HUB_TOKEN', "")
PROXY_ENV = f"-e HTTP_PROXY={os.environ.get('HTTP_PROXY', "")} -e HTTPS_PROXY={os.environ.get('HTTPS_PROXY', "")} -e NO_PROXY={os.environ.get('NO_PROXY', "")}"

logging.basicConfig(
        level=logging.DEBUG,
        format="{asctime}.{msecs:.0f} - {levelname} - {message}",
        style="{",
        datefmt="%H:%M:%S",
        handlers = [
                logging.FileHandler('results/tuner.log'),
                logging.StreamHandler()
            ]
        )


def run_docker_cmd(docker_command, is_exit=True):
    logging.debug(f"Running docker command: {docker_command}")
    try:
        process = subprocess.Popen(shlex.split(docker_command), shell=False)
        process.wait()
        #result = subprocess.run("ls", check=True, capture_output=True, text=True)
    except Exception as e:
        logging.exception(f"Error running docker cmd, exception encountered is {e}")
        if is_exit:
            sys.exit(1)
    #return result


def stop_nginx():
    docker_command1 = "docker stop nginx-lb"
    docker_command2 = "docker rm nginx-lb"
    run_docker_cmd(docker_command1)
    run_docker_cmd(docker_command2, is_exit=False)


def stop_vllm(numa_conf):
    stop_nginx()

    for i in range(len(numa_conf)):
        container_name = 'vllm'+str(i)
        docker_command1 = f"docker stop {container_name}"
        docker_command2 = f"docker rm {container_name}"
        logging.debug(f"Stopping and removing container {container_name}")
        run_docker_cmd(docker_command1)
        run_docker_cmd(docker_command2, is_exit=False)
        

    logging.info("Waiting for 15s after stopping and removing vllm containers")
    time.sleep(15)


def get_model_res_dir(model):
    m = model.replace('/', '--')
    return f"{RESULTS_DIR_BASE}/{m}"


def run_benchmark(model, token_comb, containers_conf, is_warmup):
    container_image = containers_conf['benchmark']
    concurrency = token_comb['concurrency'] * 2
    inp_tokens = token_comb['inp_tokens']
    op_tokens = token_comb['op_tokens']
    served_model_name = SERVED_MODEL_NAME
    num_prompts = concurrency * 16
    model_dir = f"{MODEL_DIR_BASE}"
    results_dir = get_model_res_dir(model) + f"/I{inp_tokens}-O{op_tokens}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file = f"result-C{concurrency}.json"
    results_file_container = f"/results/{results_file}"
    results_file_host = f"{results_dir}/{results_file}"

    docker_command = ""

    if is_warmup:
        docker_command = f"docker run -it --rm --net=host {PROXY_ENV} -v {model_dir}:/root/.cache -e HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_HUB_TOKEN} --entrypoint=python3 {container_image} /workspace/vllm/benchmarks/benchmark_serving.py --port 8000 --dataset-name random --request-rate {concurrency} --num-prompts {num_prompts} --random-input-len {inp_tokens} --random-output-len {op_tokens} --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --served-model-name {served_model_name} --metric-percentiles 50,90,99 --max-concurrency {concurrency} --model {model}"
    else:
        docker_command = f"docker run -it --rm --net=host {PROXY_ENV} -v {model_dir}:/root/.cache -v {results_dir}:/results -e HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_HUB_TOKEN} --entrypoint=python3 {container_image} /workspace/vllm/benchmarks/benchmark_serving.py --port 8000 --dataset-name random --request-rate {concurrency} --num-prompts {num_prompts} --random-input-len {inp_tokens} --random-output-len {op_tokens} --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --served-model-name {served_model_name} --metric-percentiles 50,90,99 --max-concurrency {concurrency} --save-result --result-filename {results_file_container} --model {model}"

    run_docker_cmd(docker_command)
    return results_file_host
    

def launch_nginx():
    logging.info("Launching nginx...")
    docker_command = f"docker run --rm -itd -p 8000:80 --network vllm_nginx -v {NGINX_DIR_BASE}/nginx_conf/:/etc/nginx/conf.d/ --name nginx-lb nginx-lb:latest"
    run_docker_cmd(docker_command)
    time.sleep(2)


def launch_vllm(test, numa_conf, containers_conf):
    for i, n in enumerate(numa_conf):
        node = i
        cpuset = n['cpubind'] 
        mem =  n['membind']
        model_dir = f"{MODEL_DIR_BASE}"
        container_image = containers_conf['vllm']
        dtype = test['dtype']
        served_model_name = SERVED_MODEL_NAME
        model = test['model']
        port = 8000 + node + 1
        container_name = f"vllm{node}"
        kv_cache = test['test_parameters']['kv_cache']
        node_cpus = n['node']
        compile_config = 3
        OMP_ENV = "-e KMP_BLOCKTIME=1 -e KMP_TPAUSE=0 -e KMP_SETTINGS=0 -e KMP_FORKJOIN_BARRIER_PATTERN=dist,dist -e KMP_PLAIN_BARRIER_PATTERN=dist,dist " 
        OMP_ENV += f"-e KMP_REDUCTION_BARRIER_PATTERN=dist,dist -e VLLM_V1_USE=1 -e VLLM_CPU_OMP_THREADS_BIND={cpuset}"

#        docker_command = f"docker run -d --rm {PROXY_ENV} -p {port}:8000 --cpuset-cpus={cpuset} --cpuset-mems={mem} -e HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_HUB_TOKEN} -e VLLM_CPU_KVCACHE_SPACE={kv_cache} -v {model_dir}:/root/.cache --name {container_name} --ipc=host {container_image} --trust-remote-code --device cpu --dtype {dtype} --tensor-parallel-size 1 --enforce-eager --served-model-name {served_model_name} --model {model}"
        docker_command = f"docker run -d --rm --privileged=True {PROXY_ENV} -p {port}:8000 --network vllm_nginx --cpuset-cpus={node_cpus} --cpuset-mems={mem} {OMP_ENV} -e HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_HUB_TOKEN} -e VLLM_CPU_KVCACHE_SPACE={kv_cache} -v {model_dir}:/root/.cache --name {container_name} --ipc=host {container_image} --trust-remote-code --device cpu --dtype {dtype} --tensor-parallel-size 1 --enforce-eager --served-model-name {served_model_name} --model {model} -O{compile_config}"
    
        run_docker_cmd(docker_command)
    logging.info("Waiting 60s for all VLLM containers to initialize")
    time.sleep(60)

    for i, n in enumerate(numa_conf):
        ready = False
        port = 8000 + i + 1
        for _ in range(75):
            try: 
                response = requests.get(f"http://localhost:{port}/version", timeout=2)
                if response.status_code == 200:
                    logging.info("VLLM endpoints are available")
                    ready = True
                    break
            except (requests.ConnectionError, requests.Timeout) as  e:
                logging.info("VLLM not yet initialized, retrying in 60 seconds")

            time.sleep(60)
        if not ready:
            logging.info("Exiting test, VLLM endpoints are not available. Check vllm0 container llogs")
            sys.exit(1)

    launch_nginx()


def prepare_tests(numa_conf, models_conf):
    tests = []
    os.makedirs(RESULTS_DIR_BASE)
    for m in models_conf:
        results_dir = get_model_res_dir(m['model'])
        os.makedirs(results_dir)
        e = {'model': m['model'], 
                'dtype': m['dtype'],
                'test_parameters': m['test_parameters']}
        tests.append(e)

    return tests


def run_download(numa_conf, container_image):
    #create dir for models download
    pwd = os.getcwd()
    model_dir = MODEL_DIR_BASE

    logging.info(f"Downloading models in {model_dir}...")
    #if os.path.exists(model_dir):
    #    shutil.rmtree(model_dir)
    #os.makedirs(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    docker_command = f"docker run --rm {PROXY_ENV} -e HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_HUB_TOKEN} -v {pwd}/configs:/workspace/configs -v {model_dir}:/root/.cache {container_image}"
    run_docker_cmd(docker_command)

    
def get_json(fn):
    j = None
    try:
        with open(fn, 'r') as f:
            j = json.load(f)
    except Exception as e:
        logging.error(f"Opening file {fn} failed with exception {e}", exc_info=True)
        sys.exit(1)
    return j


def get_configs():
    numa_fn = "configs/numa.json"
    models_fn = "configs/models_benchmark.json"
    containers_fn = "configs/containers.json"

    numa_conf = get_json(numa_fn)
    models_conf = get_json(models_fn)
    containers_conf = get_json(containers_fn)
    return (numa_conf, models_conf, containers_conf)


def main():
    #Read all configs
    numa_conf, models_conf, containers_conf = get_configs()
    logging.info(numa_conf)

    #Download and quantize models
    run_download(numa_conf, containers_conf['dq'])

    #Prepare tests, create results dir
    tests = prepare_tests(numa_conf, models_conf)

    for test in tests:
        #Launch vllm server for first model (mount models dir and result dir for profile)
        launch_vllm(test, numa_conf, containers_conf)

        #Launch benchmark container for different token combinations
        for token_comb in test['test_parameters']['token_combinations']:
            #Warmup iters
            #run_benchmark(test['model'], token_comb, containers_conf, True)
            res_file = run_benchmark(test['model'], token_comb, containers_conf, False)
            results = get_json(res_file)
            token_comb['p90_op_token_throughput'] = results['output_throughput']
            token_comb['p90_ttft'] = results['p90_ttft_ms']
            token_comb['p90_tpot'] = results['p90_tpot_ms']
            token_comb['p90_itl'] = results['p90_itl_ms']
            token_comb['p90_query_lat'] = results['p90_e2el_ms']
            token_comb['p90_query_tput'] = results['request_throughput']
        stop_vllm(numa_conf)

    logging.info("--- Final test summary ---")
    for test in tests:
        logging.info(f"  Model: {test['model']}")
        for token_comb in test['test_parameters']['token_combinations']:
            logging.info(f"    Inp tokens: {token_comb['inp_tokens']}, Op tokens: {token_comb['op_tokens']}, Concurrency: {token_comb['concurrency']}")
            logging.info(f"      P90 token tput: {token_comb['p90_op_token_throughput']} tokens/sec")
            logging.info(f"      P90 Time to First Token {token_comb['p90_ttft']} ms")
            logging.info(f"      P90 Time per output token {token_comb['p90_tpot']} ms")
            logging.info(f"      P90 Inter token latency {token_comb['p90_itl']} ms")
            logging.info(f"      P90 Query latency {token_comb['p90_query_lat']} ms")
            logging.info(f"      P90 Query throughput {token_comb['p90_query_tput']} queries/sec")


if __name__ == '__main__':
    main()

