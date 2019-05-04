from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    return HttpResponse("<h2>HEY! [start] {0} [end]</h2>".format(request))
