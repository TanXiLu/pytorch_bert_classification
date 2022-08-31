# coding=utf-8
import os
import torch
import torch.nn as nn
from config import ArgConfig
from utils import prepare_device, read_json
from model import BertForSequenceClassification
from transformers import BertTokenizer


def predict(model, tokenizer, text, id2label, config):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer.encode_plus(text=text,
                                       add_special_tokens=True,
                                       max_length=config.max_seq_len,
                                       truncation='longest_first',
                                       padding="max_length",
                                       return_token_type_ids=True,
                                       return_attention_mask=True,
                                       return_tensors='pt')
        inputs['token_ids'] = inputs.pop('input_ids')
        inputs['token_type_ids'] = inputs.pop('token_type_ids')
        inputs['attention_masks'] = inputs.pop('attention_mask')
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        outputs = model(inputs)
        outputs = nn.Softmax(dim=1)(outputs)

        indexes = torch.argmax(outputs, dim=1).detach().tolist()
        ratio = torch.max(outputs).tolist()

        if len(outputs) != 0:
            outputs = [id2label[str(i)] for i in indexes]
            print(outputs, ratio)
            return outputs, ratio
        else:
            return '不好意思，我没有识别出来'


if __name__ == '__main__':
    config = ArgConfig()
    # model
    device, device_ids = prepare_device(config['n_gpu'])
    model = BertForSequenceClassification(config)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    checkpoint = torch.load(os.path.join(config['save_model_dir'], config["save_model_name"]))
    model.load_state_dict(checkpoint['state_dict'])

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['bert_dir'])
    _, id2tag = read_json(config.data_path, config.label_name)
    text = "鲍勃库西奖归谁属？ NCAA最强控卫是坎巴还是弗神新浪体育讯如今，本赛季的NCAA进入到了末段，各项奖项的评选结果也即将出炉，其中评选最佳控卫的鲍勃-库西奖就将在下周最终四强战时公布，鲍勃-库西奖是由奈史密斯篮球名人堂提供，旨在奖励年度最佳大学控卫。最终获奖的球员也即将在以下几名热门人选中产生。〈〈〈 NCAA疯狂三月专题主页上线，点击链接查看精彩内容吉梅尔-弗雷戴特，杨百翰大学“弗神”吉梅尔-弗雷戴特一直都备受关注，他不仅仅是一名射手，他会用“终结对手脚踝”一样的变向过掉面前的防守者，并且他可以用任意一支手完成得分，如果他被犯规了，可以提前把这两份划入他的帐下了，因为他是一名命中率高达90%的罚球手。弗雷戴特具有所有伟大控卫都具备的一点特质，他是一位赢家也是一位领导者。“他整个赛季至始至终的稳定领导着球队前进，这是无可比拟的。”杨百翰大学主教练戴夫-罗斯称赞道，“他的得分能力毋庸置疑，但是我认为他带领球队获胜的能力才是他最重要的控卫职责。我们在主场之外的比赛(客场或中立场)共取胜19场，他都表现的很棒。”弗雷戴特能否在NBA取得成功？当然，但是有很多专业人士比我们更有资格去做出这样的判断。“我喜爱他。”凯尔特人主教练多克-里弗斯说道，“他很棒，我看过ESPN的片段剪辑，从剪辑来看，他是个超级巨星，我认为他很成为一名优秀的NBA球员。”诺兰-史密斯，杜克大学当赛季初，球队宣布大一天才控卫凯瑞-厄尔文因脚趾的伤病缺席赛季大部分比赛后，诺兰-史密斯便开始接管球权，他在进攻端上足发条，在ACC联盟(杜克大学所在分区)的得分榜上名列前茅，但同时他在分区助攻榜上也占据头名，这在众强林立的ACC联盟前无古人。“我不认为全美有其他的球员能在凯瑞-厄尔文受伤后，如此好的接管球队，并且之前毫无准备。”杜克主教练迈克-沙舍夫斯基赞扬道，“他会将比赛带入自己的节奏，得分，组织，领导球队，无所不能。而且他现在是攻防俱佳，对持球人的防守很有提高。总之他拥有了辉煌的赛季。”坎巴-沃克，康涅狄格大学坎巴-沃克带领康涅狄格在赛季初的毛伊岛邀请赛一路力克密歇根州大和肯塔基等队夺冠，他场均30分4助攻得到最佳球员。在大东赛区锦标赛和全国锦标赛中，他场均27.1分，6.1个篮板，5.1次助攻，依旧如此给力。他以疯狂的表现开始这个赛季，也将以疯狂的表现结束这个赛季。“我们在全国锦标赛中前进着，并且之前曾经5天连赢5场，赢得了大东赛区锦标赛的冠军，这些都归功于坎巴-沃克。”康涅狄格大学主教练吉姆-卡洪称赞道，“他是一名纯正的控卫而且能为我们得分，他有过单场42分，有过单场17助攻，也有过单场15篮板。这些都是一名6英尺175镑的球员所完成的啊！我们有很多好球员，但他才是最好的领导者，为球队所做的贡献也是最大。”乔丹-泰勒，威斯康辛大学全美没有一个持球者能像乔丹-泰勒一样很少失误，他4.26的助攻失误在全美遥遥领先，在大十赛区的比赛中，他平均35.8分钟才会有一次失误。他还是名很出色的得分手，全场砍下39分击败印第安纳大学的比赛就是最好的证明，其中下半场他曾经连拿18分。“那个夜晚他证明自己值得首轮顺位。”当时的见证者印第安纳大学主教练汤姆-克雷恩说道。“对一名控卫的所有要求不过是领导球队、使球队变的更好、带领球队成功，乔丹-泰勒全做到了。”威斯康辛教练博-莱恩说道。诺里斯-科尔，克利夫兰州大诺里斯-科尔的草根传奇正在上演，默默无闻的他被克利夫兰州大招募后便开始刻苦地训练，去年夏天他曾加练上千次跳投，来提高这个可能的弱点。他在本赛季与杨斯顿州大的比赛中得到40分20篮板和9次助攻，在他之前，过去15年只有一位球员曾经在NCAA一级联盟做到过40+20，他的名字是布雷克-格里芬。“他可以很轻松地防下对方王牌。”克利夫兰州大主教练加里-沃特斯如此称赞自己的弟子，“同时他还能得分，并为球队助攻，他几乎能做到一个成功的团队所有需要的事。”这其中四名球员都带领自己的球队进入到了甜蜜16强，虽然有3个球员和他们各自的球队被挡在8强的大门之外，但是他们已经表现的足够出色，不远的将来他们很可能出现在一所你熟悉的NBA球馆里。"
    predict(model, tokenizer, text, id2tag, config)
