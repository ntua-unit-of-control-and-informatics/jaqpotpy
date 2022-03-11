from pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime


RUN pip install --upgrade pip
RUN pip install jaqpotpy
RUN pip install scikit-learn==1.0.2
RUN pip install pandas
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
RUN pip install jupyterlab

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]