# accounts/models.py

from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    
    # Payment Skeleton Fields
    is_pro_member = models.BooleanField(default=False)
    subscription_expiry = models.DateField(null=True, blank=True)
    
    # Demo/Role Fields
    is_demo_user = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} Profile"

# Auto-create Profile when User is created
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.userprofile.save()