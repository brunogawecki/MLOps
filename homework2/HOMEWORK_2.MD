# Homework Assignment: Serving a Machine Learning Model as an API Service

### **Objective**

The goal of this assignment is to **serve a machine learning model as a REST or gRPC API** using one of the following model-serving frameworks:

* **BentoML**
* **NVIDIA Triton Inference Server**
* **TorchServe**
* (or any similar model-serving tool)

You must create a service that exposes an endpoint capable of receiving requests and returning predictions from your model.

**Important:**
Use the model you trained for **HOMEWORK_1.MD**.

---

## **Assignment Details**

1. **Choose a model-serving framework** (e.g., BentoML, NVIDIA Triton, TorchServe).
2. **Prepare your trained model** from **HOMEWORK_1.MD** so it can be loaded by the serving framework.
3. **Create a service** that performs inference with the model.
4. **Expose an API endpoint** accessible via `cURL`, Python `requests`, or the tool’s official client.
5. **Demonstrate the working service** by sending an example request and showing the returned prediction.

---

## **Acceptance Criteria**

### 1. **Model-Serving Configuration / Code**

* Provide the script, configuration, or command used to run the model service. (Archive it in your code repository)

  * Examples: a BentoML service file, a TorchServe `.mar` archive setup or command line, or an NVIDIA Triton model repository directory.

### 2. **Client Request Example**

Provide an example request using *one* of the following:

* `cURL`
* Python `requests`
* A framework-specific client (e.g., TritonClient for Triton)

### 3. **Working Output**

* Demonstrate that the service successfully returns a valid model prediction.

---

## **Grading**

* **Base Score: 1.0 points**

---

## **Deadline**

**The assignment is due in two weeks: 03.12.2025**

---

## **Useful Links**

* [https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Quick_Deploy/PyTorch/README.html](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Quick_Deploy/PyTorch/README.html)
* [https://docs.bentoml.com/en/latest/examples/overview.html](https://docs.bentoml.com/en/latest/examples/overview.html)
* [https://pytorch.org/serve/](https://pytorch.org/serve/)
