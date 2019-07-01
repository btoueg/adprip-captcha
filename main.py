import base64
import hashlib
import http
import os
import shutil
import sys
from collections import defaultdict, deque
from glob import glob
from numpy import ndarray

from PIL import Image
from bs4 import BeautifulSoup
from contextlib import closing
from io import BytesIO
import requests
from tqdm import tqdm


class Config:
    area_size_noise_background = 20
    area_size_noise_fill = 25
    min_expected_pixel_for_char = 100
    vertical_step = 4


def fetch(session):
    response = session.get(
        "https://www.referendum.interieur.gouv.fr/bundles/ripconsultation/securimage/securimage_show.php",
        stream=True,
    )
    if response.status_code == 200:
        response.raw.decode_content = True
        raw = response.raw.read()
        return raw


IMAGE_CODE = "\033]1337;File=name={name};inline={inline};size={size}:{base64_img}\a"

def display_image(im):
    bio = BytesIO()
    im.save(bio, 'png')
    bio.flush()
    bio.seek(0)
    print(display_image_bytes(bio.getvalue()))

def display_image_bytes(b, filename=None, inline=1):
    """
    Display the image given by the bytes b in the terminal.
    If filename=None the filename defaults to "Unnamed file".
    """
    data = {
        "name": base64.b64encode((filename or "Unnamed file").encode("utf-8")).decode(
            "ascii"
        ),
        "inline": inline,
        "size": len(b),
        "base64_img": base64.b64encode(b).decode("ascii"),
    }
    return IMAGE_CODE.format(**data)


def create_dataset(raw_cookie, predictor=None):
    C = http.cookies.SimpleCookie()
    C.load(raw_cookie)
    cookies = requests.cookies.RequestsCookieJar()
    cookies.update({k: morsel.value for k, morsel in C.items()})
    session = requests.Session()
    session.cookies = cookies
    while True:
        response = session.get(
            "https://www.referendum.interieur.gouv.fr/consultation_publique/8/A/AA"
        )
        soup = BeautifulSoup(response.text, features="html.parser")
        form__token = soup.find(attrs={"id": "form__token"}).attrs["value"]
        raw = fetch(session)
        bio = BytesIO()
        bio.write(raw)
        bio.seek(0)
        bbio, raw_letters = _process(bio)
        prediction = "".join(
            list(predictor.predict([toArray(raw_letter) for raw_letter in raw_letters]))
        )
        print("Prediction: {}".format(prediction))
        print(display_image_bytes(raw))
        captcha = input("Enter captcha:")
        data = {"form[captcha]": captcha, "form[_token]": form__token}
        response = session.post(
            "https://www.referendum.interieur.gouv.fr/consultation_publique/8/A/AA",
            data=data,
        )
        wrong_captcha = "Mauvais code, merci de réessayer" in response.text
        if not wrong_captcha:
            if prediction.upper() == captcha.upper():
                print("Oh boy oh boy! I was right! 🙌🏼")
            else:
                print("Thanks for correcting me, human! 🙇🏻")
            sha1 = hashlib.sha1()
            sha1.update(raw)
            digest = sha1.hexdigest()
            tmp_image = "./src/{digest}_{captcha}.png".format(
                digest=digest, captcha=captcha
            )
            print("I've added this captcha to our dataset 👍🏼")
            with open(tmp_image, "wb") as g:
                g.write(raw)
            if len(prediction) != len(captcha):
                print("I even missed the letter count 😓, here's why:")
                print(display_image_bytes(bbio.getvalue()))


def _process(tmp_image, expected: str = None):
    with closing(Image.open(tmp_image)) as im:
        width, height = im.size
        px = im.load()

        def star(x, y):
            return (
                (x - 1, y + 1),
                (x - 1, y),
                (x - 1, y - 1),
                (x, y + 1),
                (x, y - 1),
                (x + 1, y + 1),
                (x + 1, y),
                (x + 1, y - 1),
            )

        def plus(x, y):
            return ((x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1))

        def paint(select_predicate, get_neighbours, color_predicate, new_color):
            for x in range(width):
                for y in range(height):
                    if select_predicate(x, y):
                        # color in green if at least one neighboor is 140
                        count_blank_neighbour = 0
                        count_140_neighbour = 0
                        for (xx, yy) in get_neighbours(x, y):
                            try:
                                if px[xx, yy] == (255, 255, 255):
                                    count_blank_neighbour += 1
                                if px[xx, yy] == (140, 140, 140):
                                    count_140_neighbour += 1
                            except IndexError:
                                pass
                        if color_predicate(count_blank_neighbour, count_140_neighbour):
                            px[x, y] = new_color

        select_predicate = lambda x, y: px[x, y] in ((140, 140, 140),)
        get_neighbours = star
        color_predicate = lambda count_blank_neighbour, count_140_neighbour: count_140_neighbour <= 2
        new_color = (255, 255, 255)

        paint(select_predicate, get_neighbours, color_predicate, new_color)
        display_image(im)

        select_predicate = lambda x, y: px[x, y] not in ((140, 140, 140), (255, 255, 255),)
        get_neighbours = plus
        color_predicate = lambda count_blank_neighbour, count_140_neighbour: count_blank_neighbour == 0 and count_140_neighbour > 0
        new_color = (0, 0, 255)

        paint(select_predicate, get_neighbours, color_predicate, new_color)
        display_image(im)

        for x in range(width):
            for y in range(height):
                if px[x, y] in ((0, 0, 255),):
                    px[x, y] = (140, 140, 140)
        display_image(im)

        select_predicate = lambda x, y: px[x, y] not in ((140, 140, 140), (255, 255, 255),)
        get_neighbours = plus
        color_predicate = lambda count_blank_neighbour, count_140_neighbour: count_blank_neighbour == 0 and count_140_neighbour > 0
        new_color = (0, 0, 255)

        paint(select_predicate, get_neighbours, color_predicate, new_color)
        display_image(im)

        for x in range(width):
            for y in range(height):
                if px[x, y] in ((0, 0, 255),):
                    px[x, y] = (140, 140, 140)
        display_image(im)

        for x in range(width):
            for y in range(height):
                if px[x, y] not in ((140, 140, 140), (255, 255, 255)):
                        px[x, y] = (255, 255, 255)

        def area(queue, predicate=None):
            content = None
            while len(queue) > 0:
                (x, y) = queue.popleft()
                try:
                    px[x, y]
                except IndexError:
                    continue
                visited.add((x, y))
                if predicate(x, y):
                    if content is None:
                        content = set()
                    content.add((x, y))
                    if (x, y + 1) not in visited:
                        queue.append((x, y + 1))
                        visited.add((x, y + 1))
                    if (x, y - 1) not in visited:
                        queue.append((x, y - 1))
                        visited.add((x, y - 1))
                    if (x + 1, y) not in visited:
                        queue.append((x + 1, y))
                        visited.add((x + 1, y))
                    if (x - 1, y) not in visited:
                        queue.append((x - 1, y))
                        visited.add((x - 1, y))
            return content

        visited = set()
        pixel_to_erase = set()
        for x in range(width):
            for y in range(height):
                if (x, y) in visited:
                    continue
                content = area(
                    deque([(x, y)]), predicate=lambda x, y: px[x, y] == (140, 140, 140)
                )
                if (
                    content is not None
                    and len(content) < Config.area_size_noise_background
                ):
                    pixel_to_erase |= content

        for (x, y) in pixel_to_erase:
            px[x, y] = (255, 255, 255)
        pixel_to_erase.clear()

        visited.clear()
        pixel_to_fill = set()
        for x in range(width):
            for y in range(height):
                if (x, y) in visited:
                    continue
                content = area(
                    deque([(x, y)]), predicate=lambda x, y: px[x, y] == (255, 255, 255)
                )
                if content is not None and len(content) < Config.area_size_noise_fill:
                    pixel_to_fill |= content

        for (x, y) in pixel_to_fill:
            px[x, y] = (140, 140, 140)
        pixel_to_fill.clear()

        # bio = BytesIO()
        # bio.flush()
        # bio.seek(0)
        # im.save(bio, "png")
        # print(display_image_bytes(bio.getvalue()))
        # im.save("tmp.png", "png")
        # sys.exit()

        ###
        # verticals = set(x for x in range(width))
        # for y in range(height):
        #     verticals &= set(x for x in range(width) if px[x, y] == (255, 255, 255))
        verticals = {}
        step = Config.vertical_step
        bool_matrix = [[False] * width for _ in range(height)]
        for y in range(0, height, step):
            for x in range(width):
                vertical_segment_is_blank = {
                    px[x, min(y + iy, height - 1)] for iy in range(step)
                } == {(255, 255, 255)}
                if vertical_segment_is_blank and (
                    y == 0
                    or x == 0
                    or bool_matrix[y - 1][x]
                    or bool_matrix[y - 1][x - 1]
                ):
                    for iy in range(step):
                        bool_matrix[min(y + iy, height - 1)][x] = True
        for x in range(width - 1, -1, -1):
            xx = x
            v = []
            for y in range(height - 1, -1, -1):
                if bool_matrix[y][xx]:
                    v.append((xx, y))
                elif xx > 0 and bool_matrix[y][xx - 1]:
                    xx -= 1
                    v.append((xx, y))
                else:
                    break
            if y == 0:
                verticals[v[-1][0]] = v
        bool_matrix = [[False] * width for _ in range(height)]
        for y in range(0, height, step):
            for x in range(width - 1, -1, -1):
                vertical_segment_is_blank = {
                    px[x, min(y + iy, height - 1)] for iy in range(step)
                } == {(255, 255, 255)}
                if vertical_segment_is_blank and (
                    y == 0
                    or x == width - 1
                    or bool_matrix[y - 1][x]
                    or bool_matrix[y - 1][x + 1]
                ):
                    for iy in range(step):
                        bool_matrix[min(y + iy, height - 1)][x] = True
        for x in range(width):
            xx = x
            v = []
            for y in range(height - 1, -1, -1):
                if bool_matrix[y][xx]:
                    v.append((xx, y))
                elif xx < width - 1 and bool_matrix[y][xx + 1]:
                    xx += 1
                    v.append((xx, y))
                else:
                    break
            if y == 0:
                verticals[v[-1][0]] = v

        for i, vertical in enumerate(verticals.values()):
            for (x, y) in vertical:
                px[x, y] = (255, 0, 0)

        char_vrange = {}
        char_hrange = {}
        pixel_count_per_char = defaultdict(int)
        for y in range(height):
            char_index = -1
            count_contiguous_blank_on_horizontal = 0
            for x in range(width):
                if px[x, y] == (255, 255, 255):
                    count_contiguous_blank_on_horizontal += 1
                else:
                    count_contiguous_blank_on_horizontal = 0
                if x == 0:
                    if px[x, y] != (255, 0, 0):
                        char_index += 1
                        if char_index not in char_vrange:
                            char_vrange[char_index] = (height, -1)
                        if char_index not in char_hrange:
                            char_hrange[char_index] = (width, -1)
                if x > 0:
                    if px[x, y] != (255, 0, 0) and px[x - 1, y] == (255, 0, 0):
                        char_index += 1
                        if char_index not in char_vrange:
                            char_vrange[char_index] = (height, -1)
                        if char_index not in char_hrange:
                            char_hrange[char_index] = (width, -1)
                if char_index > -1 and px[x, y] == (140, 140, 140):
                    pixel_count_per_char[char_index] += 1
                    px[x, y] = (char_index, char_index, char_index)
                    ymin, ymax = char_vrange[char_index]
                    char_vrange[char_index] = min(ymin, y), max(ymax, y)
                    xmin, xmax = char_hrange[char_index]
                    char_hrange[char_index] = (
                        min(xmin, x - count_contiguous_blank_on_horizontal),
                        max(xmax, x - count_contiguous_blank_on_horizontal),
                    )
        bio = BytesIO()
        im.save(bio, "png")
        bio.flush()
        bio.seek(0)
        char_indexes = sorted(char_vrange.keys())
        if expected is not None:
            if len(expected) != len(
                [
                    idx
                    for idx in char_indexes
                    if pixel_count_per_char[idx] > Config.min_expected_pixel_for_char
                ]
            ):
                return bio, []

        raw_letters = []
        for char_index in char_indexes:
            if pixel_count_per_char[char_index] <= Config.min_expected_pixel_for_char:
                continue
            xmin, xmax = char_hrange[char_index]
            ymin, ymax = char_vrange[char_index]
            with closing(im.crop((xmin, ymin, xmax, ymax))) as cropped:
                px = cropped.load()
                width, height = cropped.size
                for x in range(width):
                    for y in range(height):
                        if px[x, y] != (char_index, char_index, char_index):
                            px[x, y] = (255, 255, 255)
                cropped = cropped.resize((30, 30), Image.NEAREST)
                cropped = cropped.convert("L")
                output = BytesIO()
                cropped.save(output, "png")
                output.flush()
                output.seek(0)
                raw_letters.append(output)
        return bio, raw_letters


def toArray(img_path):
    im = Image.open(img_path)
    width, height = im.size
    px = im.load()

    image = ndarray((width, height))
    for x in range(width):
        for y in range(height):
            image[x, y] = 1 if px[x, y] < 255 else 0
    im.close()
    return image.reshape((width * height))


def trained_model():
    target_equivalence = {"c", "k", "s", "u", "v", "w", "y", "z"}
    X_chars = []
    y_labels = []
    for img_path in tqdm(sorted(glob("./dst/*"))):
        name, _ = os.path.splitext(os.path.basename(img_path))
        target, digest, captcha, idx = name.split("_")
        if target in target_equivalence:
            target = target.upper()
        y_labels.append(target)
        X_chars.append(toArray(img_path))

    from time import time
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.svm import SVC

    X_train, X_test, y_train, y_test = train_test_split(
        X_chars, y_labels, test_size=0.5, random_state=42
    )

    # print("Fitting the classifier to the training set")
    # t0 = time()
    # param_grid = {
    #     "C": [1e3, 5e3, 1e4, 5e4, 1e5],
    #     "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
    # }
    # clf = GridSearchCV(
    #     SVC(kernel="rbf", class_weight="balanced"), param_grid, cv=5, iid=False
    # )
    # clf = clf.fit(X_train, y_train)
    # print("done in %0.3fs" % (time() - t0))
    # print("Best estimator found by grid search:")
    # print(clf.best_estimator_)
    # print("Test set score of SVC: {:.2f}".format(clf.score(X_test, y_test)))

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test)))
    # clf = GridSearchCV(
    #     MLPClassifier(
    #         solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
    #     ),
    #     param_grid,
    #     cv=5,
    #     iid=False,
    # )
    # clf = MLPClassifier(
    #     solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
    # )
    # clf.fit(X_train, y_train)
    # print("Test set score of MLP: {:.2f}".format(clf.score(X_test, y_test)))
    return knn


if __name__ == "__main__":
    bio, raw_letters = _process(
        "./src/9ff494fc6eb483ecf4e84973238a07982fe0b37e_z8M6BpzE5.png", "z8M6BpzE5"
    )
    print(display_image_bytes(bio.getvalue()))
    sys.exit()

    reset = True
    if reset:
        if os.path.isdir("./dst"):
            shutil.rmtree("./dst")
        source_img_paths = sorted(glob("./src/*"))
        print("{} captchas in dataset".format(len(source_img_paths)))
        count = 0
        for img_path in tqdm(source_img_paths):
            name, ext = os.path.splitext(os.path.basename(img_path))
            sha1, captcha = name.split("_")
            bio, raw_letters = _process(img_path, captcha)
            if not raw_letters:
                print(display_image_bytes(bio.getvalue()))
                count += 1
                continue
            for idx, zipped in enumerate(zip(captcha, raw_letters)):
                letter, raw_letter = zipped
                context = {
                    "letter": letter,
                    "name": name,
                    "char_index": idx,
                    "ext": ext,
                }
                filename = "./dst/{letter}_{name}_{char_index}{ext}".format(**context)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "wb") as g:
                    g.write(raw_letter.getvalue())
        print("{} captchas discarded due to preprocessing issues".format(count))
    m = trained_model()
    raw_cookie = "xtvrn=$578644$; xtan=-; xtant=1; dtCookie=1$1E0A027E8276C2B2679766B7BC9A440C; nlbi_2043128=rlPVdehtUkMbkcVMqtI6oAAAAAD7k5Hx6egciuR/0hLaI67C; A10_Insert-20480=AJAFOJAKFAAA; incap_ses_1175_2043128=hOHUQCe3Aj8e54F9jnBOEEZTGV0AAAAA8o77FIYxdV+f2XdPRi/7yg==; visid_incap_2043128=HEM248xbS76wAgzGlbkzSfKYBl0AAAAASkIPAAAAAACA2ziNAbCXZwYmJ8qVcFl1hSL9zIEbkqHZ"
    create_dataset(raw_cookie, m)
