"""CricketPrediction URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from django.conf.urls import include, url
# from calc import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^ml_index/', views.ml_index, name="ml_index"),
    url(r'^$', views.login_page, name='login_page'),
    # url(r'^index/', views.index, name='index'),
    url(r'^web_mashup/', views.web_mashup, name='web_mashup'),
    url(r'^strategy/', views.strategy, name='strategy'),
    url(r'^displayPlayer/', views.displayPlayer, name='displayPlayer'),
    # url(r'^player/', include('player.urls')),

    url(r'^winningIndia/', views.winningIndia, name='winningIndia'),
    url(r'^rohitScore/', views.rohitScore, name='rohitScore'),
    url(r'^viratScore/', views.viratScore, name='viratScore'),
    url(r'^dhoniScore/', views.dhoniScore, name='dhoniScore'),
    url(r'^jasonRoy/', views.jasonRoy, name='jasonRoy'),
    url(r'^bumrahScore/', views.bumrahScore, name='bumrahScore'),  
    url(r'^stokesScore/', views.stokesScore, name='stokesScore'),
    url(r'^chahalScore/', views.chahalScore, name='chahalScore'),



]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)