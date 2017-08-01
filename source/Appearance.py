import math
import numpy as np
from PIL import Image
from math import sqrt
from Database import Database
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.image as image
from scipy.ndimage import rotate
from collections import defaultdict

class Appearance(object):
    '''
    Handles graphics relating to the appearance of a PC.

    1	back-center
    2	back-left
    3	back-right
    4	belt-back r
    5	belt-back-l
    6	hair-back
    7   Body
    8	head
    9	face
    10	hair-main
    11	hair-facial
    12	underhelm
    13	underpants
    14	undershirt
    15	pants
    16	pants-over
    17	shirt
    18	lower-legs
    19	footware
    20	right boot
    21	left boot
    22	shoulders
    23	belt
    24	gloves
    25	neck
    26	waistcoat
    27	coat
    28	right arm
    29	left arm
    30	right glove
    31	left glove
    32	overcoat
    33	belt-left
    34	belt-right
    35	belt-front-l
    36	belt-front-rt
    37	helm
    38	ring, rt
    39	ring, lt
    40	carried rt
    41	carried lt
    42	hands
    43	gloves_actual
    44	gloves_actual
    '''
    def __init__(self, char, db, ax, specs=''):
        self.ax = ax
        self.lst_itms = [1,  # back-center
                         2,  # back-left
                         3,  # back-right
                         4,  # belt-back r
                         5,  # belt-back-l
                         33,  # belt-left
                         34,  # belt-right
                         35,  # belt-front-l
                         36,  # belt-front-rt
                         40,  # carried rt
                         41]  # carried lt
        self.lst_body = [6,  # hair-back
                         7,  # Body
                         8,  # head
                         9,  # face
                         10, # hair-main
                         11,  # hair-facial
                         42]  # hands]
        self.lst_clths = [12,  # underhelm
                          13,  # underpants
                          14,  # undershirt
                          15,  # pants
                          16,  # pants-over
                          17,  # shirt
                          18,  # lower-legs
                          19,  # footware
                          20,  # right boot
                          21,  # left boot
                          22,  # shoulders
                          23,  # belt
                          24, # gloves, assigns to 43&44
                          25,  # neck
                          26,  # waistcoat
                          27,  # coat
                          28,  # right arm
                          29,  # left arm
                          30,  # belt, ovr
                          32,  # overcoat
                          37,  # helm
                          38,  # ring, rt
                          39,  # ring, lt
                          43,  # gloves_actual as 24
                          44]  # gloves_actual as 24
        if char is not None:
            self.char = char
            self.db = Database('NeverforgedData')
            query = ('SELECT x.loc, {}, {}, equipment.color9, equipment.shoulder, '
                     .format(', '.join(['equipment.mat{}'.format(a)
                                        for a in range(1, 3)]),
                             ', '.join(['equipment.app_{}'.format(a)
                                        for a in range(10)])) +
                     ' {}, equipment.backx, equipment.backy, equipment.itmx, '
                     .format(', '.join(['equipment.itm{}'.format(a)
                                         for a in range(1, 7)])) +
                     'equipment.itmy FROM equipment JOIN ' +
                     '(SELECT loc, item FROM PC_inventory WHERE ' +
                     'character = {} '.format(self.char.id) + 'AND worn = 1) AS x ' +
                     'ON equipment._id = x.item')
            items = self.db.query(query)
            self.items = {}
            for item in items:
                self.items[item[0]] = item
            self.pheno = self.db.query('SELECT pheno FROM PC WHERE _id = {}'
                                       .format(self.char.id))[0][0]
            self.hair = self.db.query('SELECT hair_b, hair_0, hair_1 ' +
                                      'FROM PC WHERE _id={}'.format(self.char.id))[0]
            self.face = self.db.query('SELECT {} '
                                      .format(', '.join(['face{}'.format(a)
                                                         for a in range(5)])) +
                                      'FROM PC WHERE _id={}'.format(self.char.id))[0]
        else:  # from roll20
            self.char = None
            self.db = Database('NeverforgedData')
            self.specs = specs
            print(specs)
            lst_specs = specs.split('|')
            # ********* SPECS LISTED ***************
            self.pheno = lst_specs.pop(0)  # 0
            hair = []
            hair.append(self.pheno + 'hair' + lst_specs.pop(0))
            hair.append(self.pheno + 'hair' + lst_specs.pop(0))
            hair.append(self.pheno + 'hair' + lst_specs.pop(0))
            self.hair = tuple(hair)  # 1-3
            face = []
            face.append(self.pheno + 'ears' + lst_specs.pop(0))
            face.append(self.pheno + 'eyes' + lst_specs.pop(0))
            face.append(self.pheno + 'mouth' + lst_specs.pop(0))
            face.append(self.pheno + 'ebrow' + lst_specs.pop(0))
            face.append(self.pheno + 'nose' + lst_specs.pop(0))
            face.append(lst_specs.pop(0))
            self.face = tuple(face)  # 4-9
            self.c_skin = lst_specs.pop(0)  # 10
            self.c_hair = lst_specs.pop(0)  # 11
            self.c_eye = lst_specs.pop(0)  # 12
            cloths = []
            for i in range(9):
                cloths.append(lst_specs.pop(0))
            self.c_clot = tuple(cloths)  #13-21

            # item list....
            # format |loc-item|
            # create a list of tuples...
            self.items = {}
            print(len(lst_specs), lst_specs)
            for ab in lst_specs:
                if len(ab) > 1:
                    a = ab.split('-')
                    self.items
                    query = ('SELECT {}, {}, equipment.color9, equipment.shoulder, '
                             .format(', '.join(['equipment.mat{}'.format(a)
                                                for a in range(1, 3)]),
                                     ', '.join(['equipment.app_{}'.format(a)
                                                for a in range(10)])) +
                             ' {}, equipment.backx, equipment.backy, '
                             .format(', '.join(['equipment.itm{}'.format(a)
                                                 for a in range(1, 7)])) +
                            'equipment.itmx, equipment.itmy FROM equipment ' +
                            'WHERE equipment._id = {}'.format(a[1]))
                    temp = list(db.query(query))
                    self.items[int(a[0])] = tuple([a[0]] + temp)


        # now the rest....



        self.angles = {}
        # nXStart = 235, nYStart = 87, nAddFx=-14, 13,  -30, nScalex=-1,
        self.angles[2] = (235, 87, -14, 13, -30.0, -1)
        self.angles[1] = (189, 87, 0, 13, 0.0, 1)
        self.angles[3] = (143, 87, 14, 13, -30.0, 1)
        self.angles[4] = (220, 236, 0, 5, 45.0, 1)
        self.angles[5] = (154, 236, 0, 5, -45.0, 1)
        self.angles[20] = (151, 429, 0, 0, 0.0, 1)
        self.angles[21] = (221, 429, 0, 0, 0.0, 1)
        self.angles[33] = (234, 236, 1, 5, -5.0, 1)
        self.angles[34] = (142, 236, -1, 5, 5.0, 1)
        self.angles[35] = (216, 236, 0, 5, -5.0, 1)
        self.angles[36] = (158, 236, 0, 5, -5.0, 1)
        self.angles[40] = (260, 298, 0, 0, 85, 1)
        self.angles[41] = (125, 298, 0, 0, 85, -1)

        self.args = defaultdict(list)
        self.args[31] = [0,0,0,0,0,0]
        self.hide_rt = False
        self.hide_lt = False


    def show(self):
        '''
        '''
        implot = []
        for i in range(1, 45):
            if len(self.args[i]) > 0:
                draw = True
                if i == 43 and self.hide_rt == True:
                    draw = False
                if i == 42 and self.hide_lt == True:
                    draw = False
                if draw:
                    for n in range(0, len(self.args[i]), 2):
                        try:
                            implot.append(self.ax.imshow(self.args[i][n],
                                                         **self.args[i][n+1]))
                        except:
                            print(),
        if self.char is not None:
            self.ax.set_title(self.char.name)
        self.ax.text(50, -595, r'Art from $www.HeroMachine.com$')
        if self.char is None:
            # plt.savefig('../source/static/source/static/images/' + self.specs + '.png', bbox_inches='tight')
            self.ax.get_figure().savefig('../source/static/images/temp.png', bbox_inches='tight', dpi=100)
        else:
             plt.show()

    def draw_char(self):
        self.set_colors()
        self.ax.axis('off')  # no axis
        self.ax.set_xlim(0, 400)
        self.ax.set_ylim(-600, 0)
        for i in range(1, 46):
            self.draw_loc(i)

    def draw_loc(self, i):
        exf = (0, 400, -600, 0)
        if i == 24:  # gloves...S
            try:
                self.items[24]
                self.draw_clothes(24, exf)
                # make sure the hands are colored and textured correct
                lst = ['l', 'r']
                nt = 42
                for side in lst:
                    fname = ('../source/static/images/appearance/h{}hands{}.png'
                              .format(self.pheno, side))
                    self.draw_full(fname, self.c_clot[4], exf,
                                   self.items[24][1], nt)
                    nt += 1
            except:
                pass
        elif i in self.lst_body:  # we have a body-part
            self.draw_body(i, exf)
        elif i in self.lst_clths:
            self.draw_clothes(i, exf)
        elif i in self.lst_itms:
            self.draw_itms(i)

    def draw_itms(self, i):
        '''
        '''
        try:
            # if item exists and there is something in the slot...
            item = self.items[i]
        except:
            item = None
        if i != 31:
            self.args[i] = []  # reset this dict
        if item is not None:
            if np.sum([len(item[a]) for a in range(4, 14)]) >= 6:
                exf = (0, 400, -600, 0)
                self.draw_clothes(i, exf)
                item = None
        if item is not None:
            ext = None
            for itm in range(1, 7):
                if len(str(item[itm + 14])) > 0:  # we have a file...
                    fname = ('../source/static/images/items/{}.png'
                             .format(self.items[i][itm + 14]))
                    # get colors...
                    if itm <= 2:
                        clr = self.db.query('SELECT code FROM colorlists ' +
                                            'WHERE name LIKE \'%{}%\''
                                            .format(item[itm]))[0][0]
                        mat = self.items[i][itm]
                    elif itm == 3:
                        clr = self.c_clot[1]
                        mat = self.items[i][2]
                    elif itm == 4:
                        clr = self.c_clot[3]
                        mat = 'leather'
                    elif itm == 5:
                        clr = self.c_clot[2]
                        mat = 'silk'
                    elif itm == 5:
                        clr = self.c_clot[6]
                        mat = 'ash'
                    # get the image ready...
                    image = Image.open(fname).convert('RGBA')
                    im = self.set_color(image, clr, mat)  # set color
                    #rotate
                    pivot = (item[-2], item[-1])
                    if i <= 6:
                        pivot = (item[-4], item[-3])
                    w = im.size[0]
                    h = im.size[1]
                    try:
                        if fname.index('/source/static/images/items/ovr_') > 0:
                            imr = im
                            theta = 0.0
                            hn = h
                            wn = w
                    except:
                        imr = rotate(im, -self.angles[i][4])
                        theta = (-self.angles[i][4] * math.pi/180.0)
                        hn = imr.shape[0]
                        wn = imr.shape[1]
                    if ext is None:
                        # get actual shape of image...

                        # ... order is different for reasons.

                        # trig functions

                        # theta = 0
                        cs = math.cos(theta)
                        sn = math.sin(theta)
                        hyp = sqrt((w/2 - pivot[0])**2 + (h/2 - pivot[1])**2)
                        dx = hyp*sn
                        dy = hyp*(1 - cs)
                        x = self.angles[i][0]
                        y = self.angles[i][1]
                        pvx = pivot[0] - self.angles[i][5]*dx
                        pvy = pivot[1] + dy
                        # find where the corner goes...
                        xc = self.angles[i][0] - pvx
                        yc = self.angles[i][1] - pvy
                        # extent change...
                        xn = xc - wn/2 + w/2
                        yn = yc - hn/2 + h/2
                        if self.pheno == 'f':
                            xn = xn + self.angles[i][2]
                            yn = yn + self.angles[i][3]
                        ext = (xn, xn + wn, -1*(yn + hn), -1*(yn))
                        if self.angles[i][5] == -1:
                            ext = (xn + wn, xn, -1*(yn + hn), -1*(yn))
                    # shoulder stuff...
                    if i <= 3 and item[14] != 0:
                        # change this when loading back items
                        exf = (0, 400, -600, 0)
                        if itm == 1:
                            s = 'l'
                            if i == 3:
                                s = 'r'
                            fname = ('../source/static/images/clothes/{}{}shstrap.png'
                                     .format(self.pheno, s))
                            c = self.c_clot[4]
                            m = 'leather'
                            self.draw_full(fname, c, exf, m, 31, False, i)
                    try:
                        if fname.index('items/ovr_') > 0:
                            if i == 41:
                                self.hide_rt = True
                            if i == 40:
                                self.hide_lt = True
                    except:
                        pass
                    self.args[i].append(imr)
                    self.args[i].append(dict(aspect='equal', extent=ext))

    def draw_body(self, i, exf):
        '''
        Handler for drawing body areas
        6,  # hair-back  7,  # Body  8,  # head 9,  # face 10, # hair-main
        11,  # hair-facial,  42]  # hands]
        '''
        fname = ''
        if i == 6:
            if not self.has_helm():
                fname = ('../source/static/images/appearance/hair/{}.png'
                         .format(self.hair[0])
                         .replace('G', self.pheno))
                self.draw_full(fname, self.c_hair, exf, 'none', i)
        elif i == 7:
            # body...
            fname = '../source/static/images/appearance/h{}body.png'.format(self.pheno)
            self.draw_full(fname, self.c_skin, exf, 'skin', i)
        elif i == 8:
            fname = ('../source/static/images/appearance/ear/{}.png'.format(self.face[0])
                     .replace('G', self.pheno))
            self.draw_full(fname, self.c_skin, exf, 'skin', i)
            fname = ('../source/static/images/appearance/eyes/{}.png'.format(self.face[1])
                     .replace('G', self.pheno))
            self.draw_full(fname, self.c_eye, exf, 'none', i, True)
        elif i == 9:
            fname = ('../source/static/images/appearance/mouth/{}.png'.format(self.face[2])
                     .replace('G', self.pheno))
            self.draw_full(fname, self.c_skin, exf, 'skin', i)
            fname = ('../source/static/images/appearance/ebrow/{}.png'.format(self.face[3])
                     .replace('G', self.pheno))
            self.draw_full(fname, self.c_hair, exf, 'none', i, True)
            fname = ('../source/static/images/appearance/nose/{}.png'.format(self.face[4])
                     .replace('G', self.pheno))
            self.draw_full(fname, self.c_skin, exf, 'skin', i, True)
        elif i == 10:
            if not self.has_helm():
                fname = ('../source/static/images/appearance/hair/{}.png'.format(self.hair[1])
                         .replace('G', self.pheno))
                self.draw_full(fname, self.c_hair, exf, 'none', i)
        elif i == 42:  # hands!
            fname = ('../source/static/images/appearance/h{}handsr.png'.format(self.pheno))
            self.draw_full(fname, self.c_skin, exf, 'skin', i)
            fname = ('../source/static/images/appearance/h{}handsl.png'.format(self.pheno))
            self.draw_full(fname, self.c_skin, exf, 'skin', i+1)
            self.draw_loc(24)

    def draw_clothes(self, i, ext):
        '''
        '''
        # try:
            # if item exists and there is something in the slot...
        try:
            item = self.items[i]
        except:
            item = ''
        if len(item) > 0:
            mat_n = 1
            c = ''
            for n in range(10):
                if len(self.items[i][n+3]) > 0:

                    fname = ('../source/static/images/clothes/{}.png'
                             .format(self.items[i][n+3])
                             .replace('@', self.pheno))
                    try:
                        mat = self.items[i][mat_n]
                        if mat_n == 1:
                            mat_n == 2
                    except:
                        mat = 'none'
                    c = self.c_clot[n]
                    if n == 0:
                        self.draw_full(fname, c, ext, mat, i)
                    else:
                        self.draw_full(fname, c, ext, mat, i, True)
            if len(self.items[i][14]) > 0:
                # shoulder
                fname = ('../source/static/images/clothes/{}.png'
                         .format(self.items[i][14]).replace('@', self.pheno))
                self.draw_full(fname, c, ext, mat, 31, False, i)


        else:
            # draw outter hair or underwear if applicable...
            if i == 37:
                 if not self.has_helm():
                     fname = ('../source/static/images/appearance/hair/{}.png'
                              .format(self.hair[2])
                              .replace('G', self.pheno))
                     self.draw_full(fname, self.c_hair, ext, 'none', i)
            if i == 13:
                 fname = ('../source/static/images/clothes/{}braies.png'.format(self.pheno))
                 self.draw_full(fname, 'xfaf0e6', ext, 'linen', i)
            if i == 14 and self.pheno == 'f':
                 fname = ('../source/static/images/clothes/fbra.png')
                 self.draw_full(fname, 'xfaf0e6', ext, 'linen', i)

    def draw_full(self, fname, color, ext, mat, i, apnd=False, shld=None):
        '''
        '''
        if len(fname) > 0:
            im = Image.open(fname).convert('RGBA')
            img = self.set_color(im, color, mat)
            # implot = self.ax.imshow(img, aspect='equal', extent=ext)
            if shld is not None:
                idx = (shld*2) - 2
                self.args[31][idx] = img
                self.args[31][(shld*2) - 1] = dict(aspect='equal', extent=ext)
            if apnd:
                self.args[i].append(img)
                self.args[i].append(dict(aspect='equal', extent=ext))
            else:
                self.args[i] = [img, dict(aspect='equal', extent=ext)]

    def set_colors(self):
        '''
        sets color objects...
        '''
        if self.char is not None:
            self.c_skin = self.db.query('SELECT skin FROM PC WHERE _id = {}'
                                        .format(self.char.id))[0][0]
            self.c_hair = self.db.query('SELECT hair_c FROM PC WHERE _id = {}'
                                        .format(self.char.id))[0][0]
            self.c_eye = self.db.query('SELECT eye_c FROM PC WHERE _id = {}'
                                        .format(self.char.id))[0][0]
            self.c_clot =  self.db.query('SELECT {} FROM PC WHERE _id = {}'
                                        .format(', '.join(['app_{}'.format(a)
                                                           for a in range(9)]),
                                                self.char.id))[0]
            lst = list(self.c_clot)
            lst.append('xdddddd')
            self.c_clot = tuple(lst)

    def set_color(self, image, in_color, texture='none'):
        '''
        '''
        try:
            txtr = Image.open('../source/static/images/texture/' + texture +
                              '.png').convert('RGBA')
        except:
            txtr = Image.open('../source/static/images/texture/none.png').convert('RGBA')
        h = in_color.replace('x','')
        r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))
        R = []
        G = []
        B = []
        pix_old = image.load()
        pix_txt = txtr.load()
        img = Image.new( image.mode, image.size)
        pix_new = img.load()
        w=image.size[0]
        h=image.size[1]
        rav = 221
        gav = 221
        bav = 221
        for i in range(w):
          for j in range(h):
            clrs = pix_old[i, j]
            txts = pix_txt[i, j]
            ar = clrs[0]
            ag = clrs[1]
            ab = clrs[2]
            alp = clrs[3]
            if clrs[3] > 0:
                ar = clrs[0] + txts[0] - 2*rav + r
                ag = clrs[1] + txts[1] - 2*gav + g
                ab = clrs[2] + txts[2] - 2*bav + b
            pix_new[i,j] = (ar, ag, ab, alp)
        return img

    def has_helm(self):
        '''
        if wearing a helm, returns true, else false
        '''
        try:
            if len(self.items[12]) > 0:
                return True
        except:
            try:
                if len(self.items[37]) > 0:
                    return True
            except:
                pass
        return False
