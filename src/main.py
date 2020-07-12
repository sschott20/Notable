from __future__ import division
from __future__ import print_function
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess


class FilePaths:
    "filenames and paths to data"
    char_list = '/Users/sschott/Documents/GitHub/Notable/model/charList.txt'
    accuracy = '/Users/sschott/Documents/GitHub/Notable/model/accuracy.txt'
    train = '/Users/sschott/Documents/GitHub/Notable/data/'
    infer = '/Users/sschott/Documents/GitHub/Notable/data/test.png'
    corpus = '/Users/sschott/Documents/GitHub/Notable/data/corpus.txt'


def train(model, loader):
    training_steps = 0
    best_error = float('inf')
    last_improvement = 0
    stop_val = 5
    while True:
        training_steps += 1
        print('training_steps:', training_steps)

        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iter_info = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iter_info[0], '/', iter_info[1], 'Loss:', loss)

        error_rate = validate(model, loader)

        if error_rate < best_error:
            print('Character error rate improved, save model')
            best_error = error_rate
            last_improvement = 0
            model.save()
            open(FilePaths.accuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (error_rate*100.0))
        else:
            print('Character error rate not improved')
            last_improvement += 1

        if last_improvement >= stop_val:
            print(
                'No more improvement since %d training_stepss. Training stopped.' % stop_val)
            break


def validate(model, loader):
    loader.validationSet()
    num_char_wrong = 0
    num_char_total = 0
    num_word_correct = 0
    num_word_total = 0
    while loader.hasNext():
        iter_info = loader.getIteratorInfo()
        print('Batch:', iter_info[0], '/', iter_info[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_correct += 1 if batch.gtTexts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            num_char_wrong += dist
            num_char_total += len(batch.gtTexts[i])
            print('[Correct]' if dist == 0 else '[ERR:%d]' % dist, '"' +
                  batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

    error_rate = num_char_wrong / num_char_total
    word_accuracy = num_word_correct / num_word_total
    print('Character error rate: %f%%. Word accuracy: %f%%.' %
          (error_rate*100.0, word_accuracy*100.0))
    return error_rate


def infer(model, fnImg):
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--validate',  action='store_true')
    parser.add_argument('--beamsearch',  action='store_true')
    parser.add_argument('--wordbeamsearch',  action='store_true')
    parser.add_argument('--dump',  action='store_true')

    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

    if args.train or args.validate:
        loader = DataLoader(FilePaths.train, Model.batchSize,
                            Model.imgSize, Model.maxTextLen)
        char_lis
        open(FilePaths.t, 'w').write(str().join(loader.charList))

        open(FilePaths.corpus, 'w').write(str(' ').join(
            loader.trainWords + loader.validationWords))

        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate(model, loader)

    else:
        print(open(FilePaths.accuracy).read())
        model = Model(open(FilePaths.char_list).read(),
                      decoderType, mustRestore=True, dump=args.dump)
        infer(model, FilePaths.infer)


if __name__ == '__main__':
    main()
