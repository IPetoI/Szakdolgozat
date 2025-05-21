# Numerikus Radon-transzformáció vizsgálata

## Feladatkiírás

A hallgató feladata a numerikus Radon-transzformáció vizsgálata. Praktikus okoknál fogva az említett művelet implementációja során legtöbbször azt feltételezzük, hogy a transzformálandó dobozfüggvények függvények lineáris kombinációjaként előállítható.
Ugyanakkor egy aktívan kutatott témában szükséges lenne a módszer általánosítására több, a lépcsős függvényektől eltérő bázis esetére. A szakdolgozat célja legalább egy, de lehetőleg több alternatív bázisra elvégezni ezt, majd Python nyelven írt numerikus szimulációkkal demonstrálni a kidolgozott módszer helyességét. Opcionálisan további feladat lehet még az implementált algoritmusok párhuzamosítása a Numba könyvtár segítségével.

## Összefoglaló

A szakdolgozat célja, hogy egy átfogó és alapos elemzést nyújtson a Radon-transzformáció különböző implementációs lehetőségeiről és különbségeiről, valamint azok hatékonyságáról. A modern képalkotó technológiák egyik alapvető eszköze a számítógépes tomográfia, amely a Radon-transzformáción alapul. Az említett transzformáció olyan lineáris leképezés, amely egy függvény vonalintegráljait számítja ki különböző irányú egyenesek mentén. A Radon-transzformáció kulcsszerepet játszik nemcsak az orvosi képalkotásban, hanem az ipari anyagvizsgálatban és a geofizikai kutatásokban is, így széles körben alkalmazzák.
A gyakorlatban az analitikus Radon-transzformáció diszkrét közelítésével kell dolgoznunk. Ennek kétféle változatát hasonlítjuk össze a szakdolgozatban. Az egyik esetben azt feltételezzük, hogy a Radon-transzformációt dobozfüggvények lineáris kombinációján alkalmazzuk, a másik esetben pedig körfüggvények alkotta bázist feltételezünk.
Az utóbbi esetben a körök középpontja, sugara és intenzitása alapján határozzuk meg a vonalintegrálokat. Ez pontosabb rekonstrukciót tesz lehetővé akkor, ha az eredeti függvény jobban közelíthető körfüggvények lineáris kombinációjaként. Az említett megoldás részletes összehasonlításon megy keresztül a hagyományos, az eredeti függvényt dobozfüggvények lineáris kombinációjának feltételező módszerrel. Az eljárások és azok tesztelési eredményeinek elemzése rávilágít arra, hogy a körfüggvény alapú megközelítés milyen mértékben tér el a másik módszertől, illetve hogy a rekonstrukció minősége mennyire különbözik. Az összehasonlítások során a TomoPy könyvtárat is felhasználtam, ami hozzájárult az eredmények ellenőrzéséhez. Ezáltal átfogó képet kapunk arról, mennyire alkalmazható ez a megközelítés a valós képalkotási problémák megoldásában. 
A tesztképeket egy szkript állítja elő, amiben a megjelenő körök minden paramétere lebegőpontos értékként definiált, így biztosítva, hogy minél változatosabb tesztmintákat kapjunk. A képeken alkalmazott Radon-transzformáció eredményeinek különbségei vizuálisan és számszerűen is kiértékelésre kerülnek, amiket olyan mérőszámok jellemeznek, mint a Mean Squared Error (MSE), Mean Absolute Error (MAE), valamint a Maximum Absolute Error (MaxAE). Az eredmények ezen felül különbségképek és hisztogramok formájában is megtekinthetők. Ezt követően a szinogram alapján, a TomoPy csomagban implementált gridrec rekonstrukciós algoritmus (cosine szűrővel) alkalmazásával visszanyerhető az eredeti kép egy közelítése. A rekonstruált képek és az eredeti referenciaképek közti különbségek elemzésére is sor került, amely során szemléltetésre kerülnek a különböző megoldások közötti eltérések és hiba értékek.
Mivel a CPU alapú feldolgozás jelentős számításigénnyel rendelkezik, ezt elsősorban a Radon-transzformáció működésének vizualizálására és tesztelésére alkalmas. A GPU alapú megoldás, a Numba CUDA alkalmazásával, jelentősen kihasználja a párhuzamos számítás előnyeit, ami nagymértékben lerövidíti a futásidőt.


## Mappaszerkezet

```bash
.
├── radon_rect/                    # Dobozfüggvény alapú Radon-transzformáció
│   ├── cpu/                       # CPU alapú implementáció
│   ├── gpu/                       # GPU (CUDA) gyorsítás
│   └── results/                   # Eredmények (TIFF/PNG)
├── radon_circ/                    # Körfüggvény alapú Radon-transzformáció
│   ├── cpu/
│   ├── gpu/
│   └── results/
├── radon_tomopy/                  # TomoPy könyvtárból vetítő használata
│   └── results/
├── reconstruction_tomopy/         # TomoPy könyvtárból rekonstrukció használata
├── image_generator/               # Tesztkép-generáló szkriptek
│   ├── cpu/
│   │   └── results/
│   ├── gpu/
│   │   └── results/
│   └── utils/
├── test_images/                   # Generált képek
├── txt_files/                     # Körparaméterek leíró TXT fájlok
└── tests/                         # Hibamértékek, összehasonlító tesztek
    ├── reconstruction_analysis    # Rekonstruált képek tesztelése
    │   └── results/
    └── sinogram_analysis          # Szinogramok tesztelése
        └── results/

