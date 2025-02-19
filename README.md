# Implementation of the Vision Transformer

Having read through the excellent learnpytorch.io tutorials<sup>[1]</sup>, I was eager to implement Alexey Et al.<sup>[2]</sup> to completion and do as much as possible from scratch. So here we are.

Another repo will hold the implementation of the original transformer architecture<sup>[3]</sup>. I highly recommend reading the chapter on transformers in Deep Learning<sup>[4]</sup> for an introduction alongside this paper.

# Running the Code
First install dependencies

```
poetry install  # install dependencies
```

If using VSCode can just run the debug scripts in the launch.json file. Otherwise
```
poetry shell    # activate venv
python -m scripts.main --epochs 1
```

# Citations

1. https://github.com/mrdbourke/pytorch-deep-learning/

2. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, & Neil Houlsby. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.

3. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, & Illia Polosukhin. (2023). Attention Is All You Need.

4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
