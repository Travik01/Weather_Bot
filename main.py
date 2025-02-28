import asyncio
from aiogram import Bot, Dispatcher, types
import logging
from aiogram.enums import ParseMode
from aiogram.filters.command import Command
from aiogram import Router, F
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram import F
from aiogram.client.bot import DefaultBotProperties
import matplotlib.pyplot as pylab
import tqdm
import tqdm.auto
import keras
import random
import matplotlib.pyplot as plt
from aiogram.types import InputFile
from aiogram.types import FSInputFile
from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



tqdm.tqdm = tqdm.auto.tqdm


data = fashion_mnist.load_data()
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

class_names = ['Футболка / топ', "Шорты", "Свитер", "Платье",
               "Плащ", "Сандали", "Рубашка", "Кроссовок", "Сумка",
               "Ботинок"]

trainX = np.expand_dims(trainX, -1)
testX = np.expand_dims(testX, -1)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

out_fitt = []
def model_arch():
    models = Sequential()

    models.add(Conv2D(64, (5, 5),
                      padding="same",
                      activation="relu",
                      input_shape=(28, 28, 1)))
    models.add(MaxPooling2D(pool_size=(2, 2)))
    models.add(Conv2D(128, (5, 5), padding="same",
                      activation="relu"))
    models.add(MaxPooling2D(pool_size=(2, 2)))
    models.add(Conv2D(256, (5, 5), padding="same",
                      activation="relu"))
    models.add(MaxPooling2D(pool_size=(2, 2)))
    models.add(Flatten())
    models.add(Dense(256, activation="relu"))
    models.add(Dense(10, activation="softmax"))
    return models



'''model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])'''

"""
history = model.fit(
    trainX.astype(np.float32), trainy.astype(np.float32),
    epochs=10,
    steps_per_epoch=100,
    validation_split=0.33
)"""

model = keras.models.load_model("my_model.keras")
'''
results = model.evaluate(testX, testy)
print(results)'''

#обучение модели, если вы еще не обучили раскройте комменарии и закомнтируйте строчку 77
# после обучения верните все как было, так вам не придется переобучать при каждом запуске





'''
pylab.subplot(2, 2, 1)
pylab.plot( out_fitt[0])
pylab.title("обувь")

pylab.subplot(2, 2, 3)
pylab.hist(out_fitt[1])
pylab.title("")

dataframe = pd.DataFrame(out_fitt[2])
dataframe['X_по_возрастанию'] = dataframe.X.sort_values().values
dataframe['X_по_убыванию'] = dataframe.X.sort_values(ascending=False).values
dataframe.head()'''
#график с одеждой

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


import pyowm
from datetime import datetime, timedelta
from pyowm import OWM

owm1 = OWM('YOUR API')
owm = pyowm.OWM('YOUR API')
COUNTRY = 'RU'


er = ''
er3 = []
er2 = []
at = ['0', '1', '2', '3', '4', '5']
et = []#сюда надо написать города которые выхотите что бы бот использовал, пример написания : 'Tymen'
op = ['начать']
a = [1]
u = 0
logging.basicConfig(level=logging.INFO)
bot = Bot(token='YOUR API',
          default=DefaultBotProperties(parse_mode=ParseMode.HTML)) #если выдаст ошибку спустите версию aiogram до 2
dp = Dispatcher()


class O(StatesGroup):
    c = State()
    ch = State()
    chj = State()


out_fitt = []


@dp.message(F.text.lower() == "/start")
async def cmd_food(message: types.Message, state: FSMContext):
    out_fitt = []
    kb = [
        [
            types.KeyboardButton(text="начать"),

        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True
    )

    await message.answer(
        "Здраствуйте , данный бот предскажет погоду в выбраном городе и предложит вариант одежды на выход(акуратность ответа  0.8249 (0.028950)) ",
        reply_markup=keyboard)
    await state.set_state(O.c)


@dp.message(O.c, F.text.in_(op))
async def cmd_food(message: types.Message, state: FSMContext):
    await message.answer('У какого города желаете узнать погоду?(введите город России на англиском без лишних пробелов')
    await state.set_state(O.ch)


@dp.message(O.ch, F.text.in_(et))
async def food_chosen(message: types.Message, state: FSMContext):
    await state.update_data(chosen_food=message.text.lower())
    await message.answer('введите число , то через сколько дней вы хотите узнать погоду (от 0 до 5)')
    er = message.text.lower()
    await state.set_state(O.chj)


@dp.message(O.chj, F.text.in_(at))
async def food_size_chosen(message: types.Message, state: FSMContext):
    arrive = int(message.text.lower())

    def outfut(a, c, v):
        ya = False
        yc = False
        yv = False
        img = trainX
        predictions = model.predict(img)
        while (ya == False or yc == False or yv == False):
            b = random.randint(0, 9999)
            if (np.argmax(predictions[b]) == a and ya != True):
                ya = True
                out_fitt.append(img[b])

            elif (np.argmax(predictions[b]) == c and yc != True):
                yc = True
                out_fitt.append(img[b])

            elif (np.argmax(predictions[b]) == v and yv != True):
                yv = True
                out_fitt.append(img[b])

    location = er
    forecast = owm.weather_manager().weather_at_place(location + ',' + COUNTRY)
    forecast_date = datetime.now() + timedelta(days=arrive, hours=3)

    weather = forecast.weather

    description = weather.detailed_status
    clouds = weather.clouds
    temperature = weather.temperature('celsius')['temp']
    wind = weather.wind()['speed']
    try:
        rain = weather.rain['all']
    except KeyError:
        rain = 0

    await message.answer('Общее описание погоды в это время:  ')
    if (description == 'overcast clouds'):
        await message.answer("пасмурная погода")

    elif (description == 'scattered clouds'):
        await message.answer("рассеянные облака")

    elif (description == 'broken clouds'):
        await message.answer("разорванные облака")

    elif (description == 'cloudless'):
        await message.answer("безоблачная погода")

    else:
        await message.answer(description)

    if clouds < 20:
        await message.answer('Должно быть солнечно, поэтому, возможно, понадобится шляпа или солнцезащитные очки.')

    if wind > 30:
        await message.answer("Будет ветер, так что куртка может пригодиться.")
    elif wind > 10:
        await message.answer("Будет дуть легкий ветерок, так что длинные рукава могут пригодиться.")
    else:
        await message.answer("Воздух будет довольно спокойным, так что не стоит беспокоиться о ветре.")

    if temperature < 273:
        await message.answer("Будет холодно, так что наденьте теплое пальто.")
        outfut(9, 1, 4)



    elif temperature < 283:
        await message.answer("Будет холодно, поэтому не помешает надеть пальто или толстый джемпер.")
        outfut(9, 1, 2)


    elif temperature < 293:
        await message.answer("Сейчас не слишком холодно, но вы могли бы взять с собой легкий джемпер.")
        outfut(7, 1, 6)


    else:
        await message.answer("Шорты и футболка по погоде! :)")
        outfut(5, 1, 0)

    if rain == 0:
        await message.answer("Дождя не будет, так что зонт не нужен.")
    elif rain / 3 < 2.5:
        await message.answer("Будет небольшой дождь, поэтому наденьте капюшон или зонт.")
    elif rain / 3 < 7.6:
        await message.answer("Будет небольшой дождь, так что, вероятно, понадобится зонтик.")
    elif rain / 3 < 50:
        await message.answer("Будет сильный дождь, так что тебе понадобится зонт и непромокаемая куртка.")
    elif rain / 3 > 50:
        await message.answer("Будет сильный дождь, так что наденьте спасательный жилет.")

    pylab.subplot(2, 2, 1)
    plt.imshow(out_fitt[1].squeeze(), cmap=plt.cm.binary)
    plt.axis('off')
    pylab.subplot(2, 2, 3)
    plt.imshow(out_fitt[2].squeeze(), cmap=plt.cm.binary)
    plt.axis('off')
    pylab.subplot(1, 2, 2)
    plt.imshow(out_fitt[0].squeeze(), cmap=plt.cm.binary)
    plt.axis('off')
    plt.savefig('picture.png')

    kb = [
        [
            types.KeyboardButton(text="наряд"),

        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True

    )

    await message.answer(
        "нажмите 'наряд' что бы получить наряд по погоде",
        reply_markup=keyboard)


@dp.message(F.text.lower() == "наряд")
async def get_photo(message: types.Message):
    await bot.send_photo(message.chat.id, FSInputFile("picture.png"))
    kb = [
        [
            types.KeyboardButton(text="/start"),

        ],
    ]

    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True

    )

    await message.answer(
        "Вы можете начать заново, нажав на кнопку '/start' ", reply_markup=keyboard)


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())
