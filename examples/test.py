from cosyvoice.vllm import cosyvoice2

def check_tensorrt_status():
    # 检查TensorRT引擎加载状态
    if hasattr(cosyvoice2.model.flow.decoder, 'estimator'):
        estimator = cosyvoice2.model.flow.decoder.estimator
        if hasattr(estimator, 'trt_engine'):
            print("✓ TensorRT引擎已正确加载")
            print(f"✓ TensorRT并发数: {estimator.trt_context_pool.maxsize}")
        else:
            print("⚠ TensorRT引擎未正确加载")


if __name__ == '__main__':
    check_tensorrt_status()