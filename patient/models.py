# Create your models here.
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from PIL import Image
from django.urls import reverse


class Test(models.Model):
    test_type = models.CharField(max_length=100)
    eye_image = models.ImageField(default='negative.jpg', upload_to='eye_images')
    description = models.TextField()
    date_tested = models.DateTimeField(default=timezone.now)
    p_user = models.ForeignKey(User, on_delete=models.CASCADE)
    test_result = models.CharField(max_length=100,default='Negative')

    def __str__(self):
        return self.test_type

    def get_absolute_url(self):
        return reverse('test-detail', kwargs={'pk': self.pk})

    # def test_img(self):
        # imgn???


# python manage.py makemigrations
#   chk migratio folder
# python manage.py sqlmigrate patient migration_NO
# python manage.py migrate
#
# main proj
#     settings.py
#         MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
#         MEDIA_URL = '/media/'
#
#         CRISPY_TEMPLATE_PACK = 'bootstrap4'
#
#         LOGIN_REDIRECT_URL = 'patient-home'
#         LOGIN_URL = 'login'



# <img class="rounded-circle account-img" src="{{ user.profile.image.url }}">
#     main-proj -> urls.py
# from django.conf import settings
# from django.conf.urls.static import static
#
# url....
#
# if settings.DEBUG:
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


#     default-img set
# add img insie /mdeia/

#
# users -> signals.py
