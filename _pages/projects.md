---
permalink: /projects/
title: "Projects"
author_profile: true
redirect_from: 
  - /projects/
---


## **Natural Language Processing (NLP) & Large Language Models (LLMs)**  

**[Text Classification](https://github.com/ashaduzzaman-sarker/Natural-Language-Processing-NLP/tree/main/0.%20Text%20Classification):**  

- Fine-tuned **BERT-base-cased** model on the **MRPC** dataset to classify paraphrase pairs, improving model accuracy for text similarity tasks.

- Developed a sentiment analysis model by fine-tuning **DistilBERT** on the **IMDb** dataset to predict whether movie reviews are positive or negative.


**[Token Classification/Named Entity Recognition (NER)](https://github.com/ashaduzzaman-sarker/Natural-Language-Processing-NLP/tree/main/1.%20Token%20classification):**  

- Implemented token classification and named entity recognition (NER) on the CoNLL-2003 dataset using PyTorch, focusing on accurately identifying and classifying named entities in text.

- Developed a token classification model for named entity recognition on the WNUT dataset, enhancing the ability to detect and classify emerging and less common entities in diverse text.


**[Masked Language Modeling (MLM)](https://github.com/ashaduzzaman-sarker/Natural-Language-Processing-NLP/tree/main/2.%20Masked%20language%20modeling):**

- Developed and fine-tuned a **DistilBERT** model for masked language modeling on the **IMDb** dataset. Preprocessed the dataset and applied MLM techniques to enhance model's contextual understanding for sentiment analysis tasks.  

- Fine-tuned **DistilRoBERTa** on the **ELI5** dataset to perform masked language modeling, improving the model's ability to predict missing words in long-form answers. Applied robust training strategies and evaluated the model's performance using PyTorch. 

**[Machine Translation](https://github.com/ashaduzzaman-sarker/Natural-Language-Processing-NLP/tree/main/3.%20Translation):**

- Fine-tuned a Marian transformer model on the KDE4 dataset for machine translation tasks, focusing on generating accurate translations between different language pairs. Applied data preprocessing and optimized training for efficient translation performance.

- Developed and fine-tuned a T5 model on the OPUS dataset to perform sequence-to-sequence machine translation. Enhanced the model’s ability to translate multilingual text and evaluated its accuracy and efficiency using advanced techniques in PyTorch.

**[Text Summarization](https://github.com/ashaduzzaman-sarker/Natural-Language-Processing-NLP/tree/main/4.%20Summarization):**

- Fine-tuned a T5 model for text summarization on the BillSum dataset, focusing on generating concise summaries of U.S. Congressional bills. Applied transformer-based techniques for sequence-to-sequence learning, optimizing the model for efficient summarization.  

- Fine-tuned the mT5 model on Amazon Reviews to generate summaries from customer reviews. Leveraged multilingual capabilities of the model for summarizing reviews across different languages and evaluated the model’s performance using PyTorch.

**[Causal Language Modeling](https://github.com/ashaduzzaman-sarker/Natural-Language-Processing-NLP/tree/main/5.%20Casual%20Language%20modeling):**

- Fine-tuned DistilGPT2 for causal language modeling on the ELI5 dataset, focusing on generating human-like answers to complex questions. Implemented effective training strategies to improve model performance in long-form text generation tasks.  

- Fine-tuned GPT-2 on the Codeparrot dataset for code generation, enhancing the model's ability to predict and generate accurate code snippets. Applied causal language modeling techniques to train the model effectively for coding tasks.

**[Question Answering](https://github.com/ashaduzzaman-sarker/Natural-Language-Processing-NLP/tree/main/6.%20Question%20Answering):**

- Developed and fine-tuned a BERT model for question answering tasks on the SQuAD dataset. Implemented data preprocessing, model training, and evaluation techniques to enhance the model's accuracy in answering questions based on context.  

- Fine-tuned DistilBERT for question answering on the SQuAD dataset, focusing on optimizing performance while maintaining efficiency. Applied advanced techniques for model evaluation and validation to achieve high accuracy in response generation.

**[Multiple Choice](https://github.com/ashaduzzaman-sarker/Natural-Language-Processing-NLP/tree/main/7.%20Multiple%20choice):**

- Fine-tuned a BERT model for multiple-choice question answering on the SWAG dataset, focusing on predicting the most plausible continuation of a given context. Implemented data preprocessing and evaluation metrics to enhance model accuracy and robustness in understanding language semantics.


## Computer Vision

**[Image Classification](https://github.com/ashaduzzaman-sarker/Computer-Vision-CV-Projects/tree/main/0.%20Image%20Classification):**

- Developed a model employing attention mechanisms for classifying handwritten digits in the MNIST dataset, focusing on improving performance through deep multiple instance learning techniques.

- Implemented a few-shot learning approach using the Reptile algorithm to classify characters in the Omniglot dataset, demonstrating effective adaptation to new classes with minimal training examples.

- Fine-tuned a Vision Transformer (ViT) model to classify food items in images from the Food 101 dataset, leveraging transfer learning for improved accuracy in food image classification tasks.

- Applied the BiT framework for image classification tasks, demonstrating the ability to transfer knowledge from large datasets to enhance performance on target datasets.

- Explored the GCViT architecture for image classification, focusing on its ability to capture global context in visual data for improved classification accuracy.

- Developed a straightforward convolutional neural network for classifying handwritten digits in the MNIST dataset using TensorFlow, showcasing fundamental CNN techniques.

- Fine-tuned an EfficientNet model to classify dog breeds in the Stanford Dogs dataset, utilizing advanced transfer learning strategies for high accuracy.

- Built and trained a Vision Transformer model from scratch on the CIFAR-100 dataset, focusing on the unique challenges posed by diverse classes.

- Implemented involutional neural networks for image classification tasks, exploring novel architectures to enhance model performance.

- Fine-tuned Swin Transformers for various image classification tasks, leveraging their hierarchical representation capabilities for better feature extraction.

- Developed a CNN model for binary classification of cat and dog images, employing TensorFlow for effective training and evaluation.

- Implemented modern MLP architectures, including MLP Mixer, FNet, and gMLP, for classifying images in the CIFAR-100 dataset, exploring innovative approaches to image representation.

- Explored compact convolutional transformer architectures for classifying images in the CIFAR-10 dataset, focusing on efficiency and accuracy.

- Developed a CNN model for classifying pneumonia in medical images, utilizing TPU for accelerated training and improved inference speed.

- Implemented a semi-supervised learning approach using SimCLR for contrastive pretraining, enhancing image classification performance with limited labeled data.

- Investigated the ShiftViT architecture for image classification tasks, focusing on alternative mechanisms to attention for improved efficiency.

- Developed a Vision Transformer from scratch, employing innovative tokenization and attention techniques to enhance performance on smaller datasets.

- Implemented a zero-shot learning approach for image classification, demonstrating the model's ability to classify unseen categories without prior training on those classes.

**[Image Segmentation](https://github.com/ashaduzzaman-sarker/Computer-Vision-CV-Projects/tree/main/1.%20Image%20Segmentation):**

- Fine-tuned the Segment Anything Model (SAM) for various segmentation tasks using Keras and Hugging Face's Transformers library, focusing on enhancing segmentation accuracy and model adaptability.

- Developed and fine-tuned a SegFormer model on the SceneParse150 dataset, demonstrating robust performance in semantic segmentation tasks through advanced transformer-based techniques.

- Implemented the BASNet architecture for precise boundary segmentation in images, achieving high accuracy in delineating object boundaries through innovative network design.

- Explored composable fully convolutional networks for image segmentation, focusing on modular architecture design to enhance flexibility and performance in segmentation tasks.

- Developed a U-Net model for biomedical image segmentation, leveraging its encoder-decoder structure for effective feature extraction and spatial information retention.

- Implemented DeepLabV3+ for multiclass semantic segmentation tasks, utilizing atrous convolution and spatial pyramid pooling to achieve superior segmentation performance on complex datasets.

**[Object detection](https://github.com/ashaduzzaman-sarker/Computer-Vision-CV-Projects/tree/main/2.%20Object%20detection):**

- Fine-tuned the DETR model on the CPPE 5 dataset for object detection tasks using PyTorch, focusing on improving detection accuracy through advanced transformer-based techniques.

- Developed keypoint detection models utilizing transfer learning to enhance performance in detecting critical points in images.

- Implemented object detection using the RetinaNet architecture, optimizing the model for robust performance in detecting objects across various categories.

- Explored object detection with Vision Transformers (ViT), demonstrating effective application of transformer architectures in visual tasks.

- Conducted text-prompted zero-shot object detection using PyTorch, leveraging natural language prompts to identify and classify objects without prior training on specific categories.

**[Zero Shot Computer Vision](https://github.com/ashaduzzaman-sarker/Computer-Vision-CV-Projects/tree/main/3.%20Zero%20Shot%20Computer%20Vision):**

- Developed a monocular depth estimation model using PyTorch, focusing on predicting depth information from single images to enhance understanding of spatial relationships in visual data.

- Conducted text-prompted zero-shot object detection using PyTorch, leveraging natural language prompts to identify and classify objects without prior training on specific categories.

- Implemented a zero-shot image classification model in PyTorch, demonstrating the ability to classify images into unseen categories using descriptive text prompts.


## Multimodal-Vision-Language-Models

**[Image Captioning](https://github.com/ashaduzzaman-sarker/Multimodal-Vision-Language-Models/tree/main/0.%20Image%20captioning):**

- Fine-tuned the GIT image captioning model on the Pokémon BLIP dataset using PyTorch, focusing on generating accurate and contextually relevant captions for images in the dataset.

**[Document Question Answering (DocVQA)](https://github.com/ashaduzzaman-sarker/Multimodal-Vision-Language-Models/tree/main/1.%20Document%20Question%20Answering%20(DocVQA)):**

- Fine-tuned the LayoutLMv2 model for document question answering on the DocVQA dataset using PyTorch, enhancing the model's ability to understand and respond to questions based on document layout and content.

**[Visual Question Answering (VQA)](https://github.com/ashaduzzaman-sarker/Multimodal-Vision-Language-Models/tree/main/2.%20Visual%20Question%20Answering%20(VQA)):**

- Fine-tuned a Visual Question Answering (VQA) model, ViLT, on the Graphcore VQA dataset using PyTorch, improving the model's capability to answer questions about images by leveraging visual and textual information.

**[Text-to-speech (TTS)](https://github.com/ashaduzzaman-sarker/Multimodal-Vision-Language-Models/tree/main/3.%20Text-to-speech%20(TTS)):**

- Fine-tuned SpeechT5 for text-to-speech (TTS) tasks on the VoxPopuli dataset using PyTorch, enhancing the model's ability to generate natural and expressive speech from textual input.

**[Image-text-to-text](https://github.com/ashaduzzaman-sarker/Multimodal-Vision-Language-Models/tree/main/4.%20Image-text-to-text):**

- Fine-tuned SpeechT5 for text-to-speech (TTS) tasks on the VoxPopuli dataset using PyTorch, enhancing the model's ability to generate natural and expressive speech from textual input.
