from django.db import models

# Create your models here.
class Prices(models.Model):
    Open = models.IntegerField()
    High = models.IntegerField()
    Low = models.IntegerField()
    PrevClose = models.IntegerField()
    class Meta:
        db_table='Prices'