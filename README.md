# Subword Language Model for Query Auto-Completion

This is the official github repository for [Subword Language Model for Query Auto-Completion](https://arxiv.org/abs/1909.00599) (EMNLP-IJCNLP 2019).


## Dependencies
- Python 3
- PyTorch
- SentencePiece


## Preparing Data
- Dowload original AOL query log dataset: `./get_data.sh`.
  This files will be saved in `data/aol/org` directory.
- Split this data into `{train, valid, test}.{query, uid, time}.txt` by giving name tag for the split and specifying time interval of each split.
  It will be generated in the `data/aol/<tag>` directory.
  Or, you can just run `split.sh` to use a pre-determined partition setting. 
  ```
  python split.py --tag full  --train_start "2006-03-01 00:00:00" --train_end "2006-05-18 00:00:00" \
                              --valid_start "2006-05-18 00:00:00" --valid_end "2006-05-25 00:00:00" \
                              --test_start  "2006-05-25 00:00:00" --test_end  "2006-06-01 00:00:00"
  ```
- Train [SentencePiece](https://github.com/google/sentencepiece/) models (char, bpe, and unigram): `./train_spms.sh`. 
  You may change the subword vocabulary size (default: 256).


## Training a language model
```
python train.py \
    --data_dir data/aol/full \
    --spm <spm> \               # char, bpe/<vocab-size>, or unigram/<vocab-size> 
    --sample -1 0.2 \           # if spm is ungiram
    --ninp 100 \
    --nhid 600 \
    --nlayers 1 \
    --max_seq_len 40
```  

## Generating completions using a trained language model
``` 
python generate.py \
    --gen_bsz 1 \
    --beam_size 30 \
    --branching_factor 30 \
    --retrace <R> \             # for the retrace algorithm
    --nbest <n> \               # for the n-best decoding
    --do_merge \                # for marginalization
```


## Citation
If you find this work useful, please cite:
```
@article{kim2019subword,
  title={Subword Language Model for Query Auto-Completion},
  author={Kim, Gyuwan},
  journal={arXiv preprint arXiv:1909.00599},
  year={2019}
}
```


## Contact Information
Please feel free to contact Gyuwan Kim ([gyuwan.kim@navercorp.com](mailto:gyuwan.kim@navercorp.com)) if there is any question.


## License
MIT License

```
Copyright (c) 2019-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```
