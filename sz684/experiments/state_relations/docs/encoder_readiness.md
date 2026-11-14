# Text encoder readiness

Use the historical BLOOMZ text checkpoint (`bigscience/bloomz-1b7`, `4ea0141c75b2b96e17bc41f0d52c5368c879a281`, layer 19) and fixed DINOv2 vision checkpoint (`timm/vit_base_patch14_dinov2.lvd142m`, `4685c99dabffe5affac90bd99dbffd25801ae58d`, layer 10) for the first baseline.

The priority candidate is the authors' Flickr30k text JEPA trained on different captions of each image. The paper uses frozen backbone features. The authors' public model listing and Space file tree did not expose the text encoder weights, and the demo code refers to a separately packed checkpoint. Run decision: BLOOMZ is the baseline. The JEPA comparison is pending a usable checkpoint or the authors' documented CUDA training run. The matched masked-span arm uses the same Flickr recipe with the caption-views flag omitted. [Paper setup](https://arxiv.org/html/2610.39302v1#S3) · [Training recipe](https://github.com/galilai-group/prh-proof#quick-start-two-jepas-on-flickr30k-one-gpu-about-two-hours) · [Demo files](https://huggingface.co/api/spaces/galilai-group/prh_demo/tree/main?recursive=true)

When a checkpoint is available, pin its hash, tokenizer and backbone pooling. Embed paired state prompts from the existing templates on the development split and check for finite, nonzero state differences. Refit the same bridge on the same training examples and exercise retrieval on validation. Use the frozen test roster and the same edit method for the encoder comparison.

The [demo entry point](https://huggingface.co/spaces/galilai-group/prh_demo/blob/main/app.py) reads `captioner.pt`, `gallery.pt`, and example images from the deployment's `PRH_WEIGHTS` setting. I found no public MIT-States archive link there. The [paper repository](https://github.com/galilai-group/prh-proof#data) documents a different set of datasets for its experiments.

The public repository has no GitHub release. An open [checkpoint release request](https://github.com/galilai-group/prh-proof/issues/2) had no replies when checked. No author checkpoint was downloadable through the arXiv links. The similarly named Hugging Face model Zual/prh_txt was created in 2024 and cannot be attributed to this paper.
