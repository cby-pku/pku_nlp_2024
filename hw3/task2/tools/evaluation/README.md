更改utils.py 中的 决定翻译内容

```
def get_annotator_correction_prompt(data, type):
    # restriction_list = RestrictionList #### 第一步所有restrictions都是violation
    
    # TODO 在这里更改输入，如果只翻译question就只填question，如果翻译1Q1A就只填一个response
    if (type == 'translate_en_ch'):
        question = str(data['question'])
        user_prompt = TRANSLATE_EN_IN_CH_USER_PROMPT.format(
            prompt =question,
            responseA = '',
            responseB = ''
        )
        return TRANSLATE_EN_IN_CH_SYSTEM_PROMPT,user_prompt
    if (type == 'translate_ch_en'):
        question = str(data['question'])
        user_prompt = TRANSLATE_CH_IN_EN_USER_PROMPT.format(
            prompt =question,
            responseA = '',
            responseB = ''
        )
        return TRANSLATE_CH_IN_EN_SYSTEM_PROMPT,user_prompt 
    
    else:
        raise RuntimeError("not implemented type")
```


/translate_ch_en.sh 中译英
/translate_en_ch.sh 英译中
