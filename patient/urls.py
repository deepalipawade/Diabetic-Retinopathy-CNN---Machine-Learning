from django.urls import path
from . import views
from .views import (
    TestListView,
    TestDetailView,
    TestCreateView,
    TestUpdateView,
    TestDeleteView
)
urlpatterns = [
    path('', TestListView.as_view(), name='patient-home'),
    path('about/', views.about, name='patient-about'),
    path('preprocess/', views.preprocess, name='preprocess'),
    path('output/', views.output, name='patient-output'),
    path('test/<int:pk>/', TestDetailView.as_view(), name='test-detail'),
    path('test/new/', TestCreateView.as_view(), name='test-create'),
    path('test/level/', views.leveltest, name='test-level'),
    path('test/<int:pk>/update/', TestUpdateView.as_view(), name='test-update'),
    path('test/<int:pk>/delete/', TestDeleteView.as_view(), name='test-delete'),
]


# see latest tutorial for create update for function call
