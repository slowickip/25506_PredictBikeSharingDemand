# **Przewidywanie poziomu zatłoczenia ruchu drogowego w miastach**

## Opis tematu i problemu biznesowego/technicznego

### Temat projektu
Celem projektu jest stworzenie modelu predykcyjnego, który przewiduje poziom zatłoczenia ruchu drogowego w wybranych miastach na podstawie danych historycznych. Model pozwala zidentyfikować wzorce zatłoczenia i wspierać użytkowników, takich jak kierowcy oraz zarządcy infrastruktury, w planowaniu tras i zarządzaniu ruchem.

### Problem biznesowy
Zatłoczenie na drogach jest powszechnym problemem, prowadzącym do zwiększonych kosztów ekonomicznych, spadku jakości życia oraz wzrostu emisji zanieczyszczeń. Przewidywanie natężenia ruchu drogowego w określonych godzinach i dniach tygodnia może pomóc w lepszym planowaniu tras, unikaniu korków oraz wspieraniu służb miejskich w zarządzaniu infrastrukturą drogową i reagowaniu na zdarzenia drogowe.

---

## Źródło danych, charakterystyka i uzasadnienie wyboru

### Źródło danych
Dane zostały pobrane z [Kaggle: TomTom Traffic Data: 55 Countries, 387 Cities](https://www.kaggle.com/datasets/bwandowando/tomtom-traffic-data-55-countries-387-cities/data). Jest to obszerny zbiór danych obejmujący dane o natężeniu ruchu w różnych miastach na całym świecie.

### Charakterystyka danych
Zbiór danych zawiera informacje o poziomach zatłoczenia ruchu drogowego, w tym opóźnieniach i liczbie zatorów, dla 387 miast w 55 krajach. Każdy rekord zawiera następujące atrybuty:

- `Country`: kraj, w którym zebrano dane,
- `City`: miasto, w którym zebrano dane,
- `UpdateTimeUTC`: czas aktualizacji danych (w UTC),
- `JamsDelay`: opóźnienie spowodowane zatorami (w minutach),
- `TrafficIndexLive`: bieżący wskaźnik zatłoczenia,
- `JamsLengthInKms`: łączna długość zatorów drogowych (w kilometrach),
- `JamsCount`: liczba zatorów drogowych,
- `TrafficIndexWeekAgo`: wskaźnik zatłoczenia sprzed tygodnia,
- `UpdateTimeUTCWeekAgo`: czas aktualizacji danych sprzed tygodnia,
- `TravelTimeLivePer10KmsMins`: bieżący czas przejazdu na 10 km (w minutach),
- `TravelTimeHistoricPer10KmsMins`: historyczny czas przejazdu na 10 km (w minutach),
- `MinsDelay`: całkowite opóźnienie w minutach.

### Uzasadnienie wyboru
Dane te umożliwiają analizę wpływu różnych czynników zewnętrznych, takich jak pora dnia i dzień tygodnia, na zatłoczenie w miastach. Zawierają wystarczającą liczbę rekordów i atrybutów numerycznych, które pozwalają na stworzenie dokładnego modelu predykcyjnego. Informacje historyczne umożliwiają również ocenę, jak poziom zatłoczenia zmienia się w czasie.

---

## Cele projektu

1. **Analiza wzorców ruchu drogowego**: Zrozumienie, które czynniki mają największy wpływ na zatłoczenie dróg w różnych miastach, np. długość zatorów, dzień tygodnia.
2. **Budowa modelu predykcyjnego**: Stworzenie modelu, który przewiduje poziom zatłoczenia ruchu drogowego, co może pomóc w zarządzaniu ruchem.
3. **Wizualizacja i raportowanie wyników**: Stworzenie intuicyjnych wizualizacji pokazujących obecne oraz przewidywane poziomy zatłoczenia w wybranych miastach.

---

## Struktura pracy nad modelem

1. **Przetwarzanie danych**:
   - **Pobieranie danych z Kaggle**: zaimportowanie danych do środowiska pracy.
   - **Obróbka i przygotowanie danych**: czyszczenie danych, uzupełnianie brakujących wartości oraz usuwanie anomalii.

2. **Eksploracja i analiza danych (EDA)**:
   - **Przeprowadzenie eksploracyjnej analizy danych (EDA)**: wizualizacja poziomów zatłoczenia w zależności od długości zatorów, liczby zatorów, opóźnień i pory dnia.
   - **Wykrywanie braków i wstępne przetwarzanie**: identyfikacja brakujących wartości i uzupełnianie danych.

3. **Trenowanie modelu**:
   - **Dobór modelu ML**: wybór odpowiedniego modelu predykcyjnego (np. regresja liniowa, lasy losowe).
   - **Dostosowanie modelu do problemu predykcyjnego**: trenowanie i optymalizacja modelu na danych treningowych.

4. **Walidacja i testowanie**:
   - **Walidacja modelu na danych testowych**: ocena modelu na zbiorze testowym za pomocą odpowiednich metryk (np. RMSE, MAE).
   - **Monitorowanie wyników**: analiza skuteczności modelu i jego stabilności.

5. **Dokształcanie modelu**:
   - **Dalsze trenowanie na nowych danych**: dostosowanie modelu do nowych danych w celu poprawy dokładności predykcji.

6. **Publikacja i wdrożenie**:
   - **Przygotowanie modelu do wdrożenia**: zapewnienie stabilności modelu w produkcyjnym środowisku.
   - **Konteneryzacja (Docker)**: umieszczenie modelu w kontenerze, aby ułatwić wdrożenie.

7. **Prezentacja i raport końcowy**:
   - **Przygotowanie prezentacji wyników**: stworzenie wizualizacji i dashboardów pokazujących wyniki modelu.
   - **Opracowanie raportu końcowego**: dokumentacja wyników projektu i analiza końcowa.


### Analiza automatycznego raportu

#### Podstawowe informacje o danych
- **Liczba wierszy**: 429544
- **Liczba kolumn**: 12
- **Typy danych**: 8 kolumn typu `float64`, 4 kolumny typu `object`
- **Brakujące wartości**: Brak brakujących wartości

#### Rozkłady zmiennych numerycznych
- **JamsDelay**: Średnia wartość to 185.29, z dużym odchyleniem standardowym 585.23. Wartości wahają się od 0 do 15568.6.
- **TrafficIndexLive**: Średnia wartość to 0.70, z odchyleniem standardowym 1.10. Wartości wahają się od -11.70 do 20.50.
- **JamsLengthInKms**: Średnia wartość to 0.70, z odchyleniem standardowym 1.10. Wartości wahają się od -11.70 do 20.50.
- **JamsCount**: Średnia wartość to 0.70, z odchyleniem standardowym 1.10. Wartości wahają się od -11.70 do 20.50.
- **TrafficIndexWeekAgo**: Średnia wartość to 0.70, z odchyleniem standardowym 1.10. Wartości wahają się od -11.70 do 20.50.
- **TravelTimeLivePer10KmsMins**: Średnia wartość to 0.70, z odchyleniem standardowym 1.10. Wartości wahają się od -11.70 do 20.50.
- **TravelTimeHistoricPer10KmsMins**: Średnia wartość to 0.70, z odchyleniem standardowym 1.10. Wartości wahają się od -11.70 do 20.50.
- **MinsDelay**: Średnia wartość to 0.70, z odchyleniem standardowym 1.10. Wartości wahają się od -11.70 do 20.50.

#### Korelacje
- **JamsCount** jest wysoko skorelowany z **JamsDelay** i 6 innymi polami.
- **JamsDelay** jest wysoko skorelowany z **JamsCount** i 6 innymi polami.
- **JamsLengthInKms** jest wysoko skorelowany z **JamsCount** i 5 innymi polami.
- **MinsDelay** jest wysoko skorelowany z **JamsCount** i 5 innymi polami.
- **TrafficIndexLive** jest wysoko skorelowany z **JamsCount** i 6 innymi polami.
- **TrafficIndexWeekAgo** jest wysoko skorelowany z **JamsCount** i 6 innymi polami.
- **TravelTimeHistoricPer10KmsMins** jest wysoko skorelowany z **JamsCount** i 4 innymi polami.
- **TravelTimeLivePer10KmsMins** jest wysoko skorelowany z **JamsCount** i 6 innymi polami.

#### Wartości unikalne
- **TravelTimeLivePer10KmsMins** ma unikalne wartości.
- **TravelTimeHistoricPer10KmsMins** ma unikalne wartości.
- **MinsDelay** ma unikalne wartości.

#### Wartości zerowe
- **JamsDelay** ma 100348 (23.4%) wartości zerowych.
- **TrafficIndexLive** ma 86114 (20.0%) wartości zerowych.
- **JamsLengthInKms** ma 102197 (23.8%) wartości zerowych.
- **JamsCount** ma 100348 (23.4%) wartości zerowych.
- **TrafficIndexWeekAgo** ma 84268 (19.6%) wartości zerowych.

### Notatki i obserwacje
1. **Wysoka korelacja**: Wiele zmiennych jest wysoko skorelowanych z **JamsCount**, co może sugerować, że te zmienne są ze sobą powiązane i mogą wpływać na siebie nawzajem.
2. **Wartości zerowe**: Znaczna liczba wartości zerowych w zmiennych takich jak **JamsDelay**, **TrafficIndexLive**, **JamsLengthInKms**, **JamsCount** i **TrafficIndexWeekAgo**. Może to wskazywać na brak ruchu lub brak danych w tych okresach.
3. **Wartości unikalne**: Zmienne **TravelTimeLivePer10KmsMins**, **TravelTimeHistoricPer10KmsMins** i **MinsDelay** mają unikalne wartości, co może być istotne przy analizie szczegółowej.

---

## Leaderboard

| Model                  | Test Score | Validation Score | Evaluation Metric | Prediction Time (Test) | Prediction Time (Validation) | Fit Time   |
|------------------------|------------|------------------|-------------------|------------------------|------------------------------|------------|
| CatBoost               | 0.61196    | 0.59605          | r2                | 3.6373                 | 0.0852                       | 3579.8433  |
| WeightedEnsemble_L2    | 0.61196    | 0.59605          | r2                | 3.7040                 | 0.0861                       | 3579.9645  |
| LightGBM               | -0.86334   | -0.89146         | r2                | 0.1987                 | 0.0062                       | 8.2212     |
| LightGBMXT             | -0.91043   | -0.95270         | r2                | 0.2758                 | 0.0058                       | 8.2586     |

---

## Wnioski

1. **Najlepszy model**: Model CatBoost osiągnął najlepsze wyniki zarówno na zbiorze testowym, jak i walidacyjnym, co czyni go najbardziej odpowiednim modelem do przewidywania poziomu zatłoczenia ruchu drogowego.
2. **Wysoka korelacja zmiennych**: Wiele zmiennych jest wysoko skorelowanych z liczbą zatorów drogowych, co sugeruje, że te zmienne są ze sobą powiązane i mogą wpływać na siebie nawzajem.
3. **Wartości zerowe**: Znaczna liczba wartości zerowych w zmiennych takich jak opóźnienia spowodowane zatorami i bieżący wskaźnik zatłoczenia może wskazywać na brak ruchu lub brak danych w tych okresach.
4. **Wartości unikalne**: Zmienne dotyczące czasu przejazdu na 10 km mają unikalne wartości, co może być istotne przy analizie szczegółowej.