# Winoground

## Dataset Download

The data can be loaded using Huggingface datasets:

```
from datasets import load_dataset
examples = load_dataset('facebook/winoground', use_auth_token=<YOUR USER ACCESS TOKEN>)
```

You can get the user access token by following these steps:

1. log into your Hugging Face account
2. click on your profile picture
3. click "Settings"
4. click "Access Tokens"
5. generate an access token

To easily view examples from Winoground, follow [this](https://huggingface.co/spaces/CVPR/winoground-explorer) link.
