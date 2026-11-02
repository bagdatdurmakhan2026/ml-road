import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    Считает матрицу ошибок:
    [[TN, FP],
     [FN, TP]]
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    return np.array([[tn, fp], [fn, tp]])

def classification_metrics(y_true, y_pred):
    """
    Возвращает ключевые метрики качества классификации.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    # Защита от деления на ноль
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    return {
        "Confusion Matrix": cm,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }
if __name__ == "__main__":
    # Тестовые данные (y_true - реальные метки, y_pred - предсказания модели)
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    results = classification_metrics(y_true, y_pred)

    print("=== ТЕСТ МЕТРИК КЛАССИФИКАЦИИ ===")
    print("Confusion Matrix:\n", results["Confusion Matrix"])
    print(f"Accuracy:  {results['Accuracy']:.4f}")
    print(f"Precision: {results['Precision']:.4f}")
    print(f"Recall:    {results['Recall']:.4f}")
    print(f"F1-Score:  {results['F1-Score']:.4f}")
    
# #
# Accuracy (Точность в целом): Простая доля правильных ответов. Ловушка: она бесполезна на несбалансированных данных (например, если из 1000 транзакций мошеннических всего 5, модель может просто всегда говорить «все транзакции честные» и получить Accuracy 99.5%, но при этом не поймать ни одного мошенника).

# Confusion Matrix (Матрица ошибок): Таблица 2х2, которая показывает все 4 исхода (верно/неверно угаданные нули и единицы). Без нее ты вообще не понимаешь, где именно ошибается твоя модель.

# Precision (Точность): Из всех объектов, которые модель назвала положительными (1), сколько реально оказались положительными?

# Пример: Спам-фильтр. Если Precision низкий, то важные рабочие письма будут улетать в спам (много ложных срабатываний — False Positive).

# Recall (Полнота): Из всех реально существующих положительных объектов в датасете, сколько наша модель вообще смогла найти?

# Пример: Диагностика рака. Если Recall низкий, значит больным людям ошибочно сказали, что они здоровы (пропустили болезнь — False Negative). Это худший сценарий.

# F1-Score: Среднее гармоническое между Precision и Recall. Нужен, когда нам нужен баланс между точностью и полнотой.





#