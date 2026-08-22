"""Long-term memory system tests.

The full ``chat.tests`` suite runs against the project's PostgreSQL
configuration. This module is a focused SQLite-in-memory variant
specifically for the long-term memory system, so the suite stays usable
on machines that do not have a local PostgreSQL reachable.

The tests here exercise:

- ``CharacterMemoryItem`` model + ``MemoryAuditLog`` audit trail.
- ``MaxLengthValidator(200)`` enforcement on ``description``.
- ``MemoryManager`` CRUD round-trips + history bookkeeping.
- REST surface ``/api/characters/{id}/memory[/...]`` for snapshot
  retrieval, CRUD, merge, and wipe.
- ``PATCH /api/sessions/{id}`` toggling ``is_private_mode``.
- ``sync_long_term_memory`` Celery task gating (private mode + user
  allow + happy-path against a mocked LLM that returns a canned
  ``create_memory`` tool call).
"""
import json
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase, override_settings


SQLITE_TEST_DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    },
}


@override_settings(DATABASES=SQLITE_TEST_DATABASES)
@override_settings(DEV_AUTO_LOGIN_ENABLED=False)
@override_settings(CELERY_TASK_ALWAYS_EAGER=True)
class CharacterMemoryModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='memory-owner', password='password123')
        from chat.models import Character
        self.character = Character.objects.create(
            created_by=self.user,
            name='Memory Character',
            avatar_url='',
            description='A character used for memory model tests.',
            personality='Calm',
            appearance='Grey coat',
            scenario='Archive',
            example_dialogue='',
            affiliation='Lab',
            tags=['memory'],
        )

    def test_create_item_persists_row_and_writes_audit_log(self):
        from chat.models import MemoryAuditAction, MemoryAuditLog, MemoryAuditSource
        from chat.memory.manager import MemoryManager

        manager = MemoryManager(self.character)
        item = manager.create_item(section='identity', description='Prefers tea over coffee.')

        self.assertTrue(item.short_id)
        self.assertEqual(item.description, 'Prefers tea over coffee.')
        self.assertEqual(item.section, 'identity')

        rows = list(
            MemoryAuditLog.objects.filter(character=self.character).order_by('created_at', 'id')
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].action, MemoryAuditAction.CREATE)
        self.assertEqual(rows[0].after_description, 'Prefers tea over coffee.')
        self.assertEqual(rows[0].before_description, '')
        self.assertEqual(rows[0].source, MemoryAuditSource.CELERY_WORKER)
        self.assertEqual(rows[0].entry_short_id, item.short_id)

    def test_create_item_rejects_oversized_description(self):
        from chat.memory.manager import MemoryManager

        manager = MemoryManager(self.character)
        with self.assertRaises(ValueError) as context:
            manager.create_item(section='identity', description='x' * 201)
        self.assertIn('exceeds 200 characters', str(context.exception))

    def test_create_item_rejects_empty_section_or_description(self):
        from chat.memory.manager import MemoryManager

        manager = MemoryManager(self.character)
        with self.assertRaises(ValueError):
            manager.create_item(section='', description='something')
        with self.assertRaises(ValueError):
            manager.create_item(section='identity', description='')

    def test_update_item_writes_history_and_audit_log(self):
        from chat.models import MemoryAuditAction, MemoryAuditLog
        from chat.memory.manager import MemoryManager

        manager = MemoryManager(self.character)
        item = manager.create_item(section='work', description='Writes in Clojure.')

        manager.update_item(
            short_id=item.short_id,
            description='Uses Clojure at work.',
            reason='Refined wording.',
            source_message=None,
        )
        item.refresh_from_db()
        self.assertEqual(item.description, 'Uses Clojure at work.')
        self.assertEqual(len(item.description_history or []), 1)
        history_entry = item.description_history[0]
        self.assertEqual(history_entry['old_desc'], 'Writes in Clojure.')
        self.assertEqual(history_entry['new_desc'], 'Uses Clojure at work.')
        self.assertEqual(history_entry['reason'], 'Refined wording.')

        audit_rows = list(MemoryAuditLog.objects.filter(character=self.character).order_by('created_at', 'id'))
        self.assertEqual([row.action for row in audit_rows], [MemoryAuditAction.CREATE, MemoryAuditAction.UPDATE])
        update_row = audit_rows[-1]
        self.assertEqual(update_row.before_description, 'Writes in Clojure.')
        self.assertEqual(update_row.after_description, 'Uses Clojure at work.')

    def test_merge_items_combines_history_and_deletes_secondary(self):
        from chat.models import CharacterMemoryItem
        from chat.memory.manager import MemoryManager

        manager = MemoryManager(self.character)
        primary = manager.create_item(section='taste', description='Likes jazz.')
        secondary = manager.create_item(section='taste', description='Dislikes cold brew.')

        merged = manager.merge_items(
            id1=primary.short_id,
            id2=secondary.short_id,
            content='Likes jazz, dislikes cold brew.',
            section='taste',
            reason='Merged two taste entries.',
        )

        self.assertEqual(merged.short_id, primary.short_id)
        self.assertEqual(merged.description, 'Likes jazz, dislikes cold brew.')
        self.assertFalse(CharacterMemoryItem.objects.filter(short_id=secondary.short_id).exists())

        actions = list(
            self.character.memory_audit_log.order_by('created_at', 'id').values_list('action', flat=True)
        )
        self.assertEqual(actions, ['create', 'create', 'merge'])

    def test_delete_item_writes_audit_log_preserving_before_description(self):
        from chat.models import CharacterMemoryItem, MemoryAuditAction
        from chat.memory.manager import MemoryManager

        manager = MemoryManager(self.character)
        item = manager.create_item(section='identity', description='Allergic to shellfish.')

        manager.delete_item(short_id=item.short_id, reason='User removed this entry.')

        self.assertFalse(CharacterMemoryItem.objects.filter(short_id=item.short_id).exists())
        delete_audit = self.character.memory_audit_log.filter(action=MemoryAuditAction.DELETE).get()
        self.assertEqual(delete_audit.before_description, 'Allergic to shellfish.')


@override_settings(DATABASES=SQLITE_TEST_DATABASES)
@override_settings(DEV_AUTO_LOGIN_ENABLED=False)
class CharacterMemoryRestTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='rest-owner', password='password123')
        self.other_user = User.objects.create_user(username='rest-other', password='password123')
        self.client.force_login(self.user)
        from chat.models import Character
        self.character = Character.objects.create(
            created_by=self.user,
            name='REST Memory Character',
            avatar_url='',
            description='Used for memory REST tests.',
            personality='Quiet',
            appearance='Brown coat',
            scenario='Office',
            example_dialogue='',
            affiliation='Keepers',
            tags=['rest'],
        )
        self.other_character = Character.objects.create(
            created_by=self.other_user,
            name='Other Character',
            avatar_url='',
            description='Owned by another user.',
            personality='Other',
            appearance='Red coat',
            scenario='Office',
            example_dialogue='',
            affiliation='Other',
            tags=['other'],
        )

    def test_get_memory_returns_empty_snapshot(self):
        response = self.client.get(f'/api/characters/{self.character.id}/memory/')
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload['sections'], [])
        self.assertEqual(payload['count'], 0)
        self.assertIn('Long-Term Memory (User Model)', payload['wiki_markdown'])

    def test_post_creates_memory_entry_with_user_edit_source(self):
        response = self.client.post(
            f'/api/characters/{self.character.id}/memory/',
            data=json.dumps({'section': 'work', 'description': 'Senior backend engineer.'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload['section'], 'work')
        self.assertEqual(payload['description'], 'Senior backend engineer.')
        self.assertTrue(payload['short_id'])

        from chat.models import MemoryAuditSource
        sources = list(
            self.character.memory_audit_log.values_list('source', flat=True)
        )
        self.assertEqual(sources, [MemoryAuditSource.USER_EDIT])

    def test_post_rejects_oversized_description(self):
        response = self.client.post(
            f'/api/characters/{self.character.id}/memory/',
            data=json.dumps({'section': 'work', 'description': 'x' * 201}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('description', response.json())

    def test_patch_updates_entry_description(self):
        self.client.post(
            f'/api/characters/{self.character.id}/memory/',
            data=json.dumps({'section': 'work', 'description': 'Junior engineer.'}),
            content_type='application/json',
        )
        items = list(self.character.memory_items.all())
        self.assertEqual(len(items), 1)

        response = self.client.patch(
            f'/api/characters/{self.character.id}/memory/{items[0].short_id}/',
            data=json.dumps({'description': 'Senior engineer.'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['description'], 'Senior engineer.')

    def test_delete_removes_entry_and_returns_description(self):
        self.client.post(
            f'/api/characters/{self.character.id}/memory/',
            data=json.dumps({'section': 'work', 'description': 'Disposable note.'}),
            content_type='application/json',
        )
        items = list(self.character.memory_items.all())
        response = self.client.delete(
            f'/api/characters/{self.character.id}/memory/{items[0].short_id}/',
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['description'], 'Disposable note.')
        self.assertFalse(self.character.memory_items.filter(short_id=items[0].short_id).exists())

    def test_merge_two_entries_keeps_primary_and_deletes_secondary(self):
        for desc in ('Loves cats.', 'Loves cats and dogs.'):
            self.client.post(
                f'/api/characters/{self.character.id}/memory/',
                data=json.dumps({'section': 'taste', 'description': desc}),
                content_type='application/json',
            )
        ids = [item.short_id for item in self.character.memory_items.order_by('created_at', 'id')]
        self.assertEqual(len(ids), 2)
        response = self.client.post(
            f'/api/characters/{self.character.id}/memory/merge/',
            data=json.dumps({
                'id1': ids[0],
                'id2': ids[1],
                'content': 'Loves cats and dogs.',
                'section': 'taste',
            }),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.character.memory_items.count(), 1)
        self.assertEqual(
            self.character.memory_items.first().description,
            'Loves cats and dogs.',
        )

    def test_delete_wipe_returns_deleted_count(self):
        self.client.post(
            f'/api/characters/{self.character.id}/memory/',
            data=json.dumps({'section': 'work', 'description': 'Stays for wipe.'}),
            content_type='application/json',
        )
        response = self.client.delete(f'/api/characters/{self.character.id}/memory/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['deleted'], 1)
        self.assertEqual(self.character.memory_items.count(), 0)

    def test_memory_routes_for_unowned_character_return_404(self):
        response = self.client.get(f'/api/characters/{self.other_character.id}/memory/')
        self.assertEqual(response.status_code, 404)


@override_settings(DATABASES=SQLITE_TEST_DATABASES)
@override_settings(DEV_AUTO_LOGIN_ENABLED=False)
class ChatSessionPrivateModeTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='private-owner', password='password123')
        self.client.force_login(self.user)
        from chat.models import Character, ChatSession
        self.character = Character.objects.create(
            created_by=self.user,
            name='Private Character',
            avatar_url='',
            description='Used for private-mode session tests.',
            personality='Quiet',
            appearance='Grey hoodie',
            scenario='Home office',
            example_dialogue='',
            affiliation='Lab',
            tags=['private'],
        )
        self.session = ChatSession.objects.create(
            user=self.user,
            character=self.character,
            title='Private Session',
        )

    def test_default_is_private_mode_is_false(self):
        response = self.client.get(f'/api/sessions/{self.session.id}/')
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['is_private_mode'])

    def test_patch_with_is_private_mode_toggles_boolean(self):
        response = self.client.patch(
            f'/api/sessions/{self.session.id}/',
            data=json.dumps({'is_private_mode': True}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['is_private_mode'])

        self.session.refresh_from_db()
        self.assertTrue(self.session.is_private_mode)

    def test_patch_accepts_string_truthy_value(self):
        response = self.client.patch(
            f'/api/sessions/{self.session.id}/',
            data=json.dumps({'is_private_mode': 'true'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['is_private_mode'])

    def test_patch_with_only_title_does_not_clobber_private_mode(self):
        self.session.is_private_mode = True
        self.session.save(update_fields=['is_private_mode', 'updated_at'])

        response = self.client.patch(
            f'/api/sessions/{self.session.id}/',
            data=json.dumps({'title': 'Renamed'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['title'], 'Renamed')
        self.assertTrue(response.json()['is_private_mode'])

        # Verify the DB row also kept the flag. Without this, a future
        # refactor that adds `update_fields=('title','updated_at')` to the
        # serializer.save() call would silently drop the flag while still
        # passing the response-body assertion above.
        self.session.refresh_from_db()
        self.assertTrue(self.session.is_private_mode)
        self.assertEqual(self.session.title, 'Renamed')


@override_settings(DATABASES=SQLITE_TEST_DATABASES)
@override_settings(DEV_AUTO_LOGIN_ENABLED=False)
class SyncLongTermMemoryTaskTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='sync-owner', password='password123')
        from chat.models import Character, ChatSession, Message
        self.character = Character.objects.create(
            created_by=self.user,
            name='Sync Character',
            avatar_url='',
            description='Used for Celery sync tests.',
            personality='Calm',
            appearance='Lab coat',
            scenario='Lab',
            example_dialogue='',
            affiliation='Lab',
            tags=['sync'],
        )
        self.session = ChatSession.objects.create(
            user=self.user,
            character=self.character,
            title='Sync Session',
        )
        self.user_message = Message.objects.create(
            chat_session=self.session,
            role='user',
            content='Tell me about your book club.',
        )
        self.assistant_message = Message.objects.create(
            chat_session=self.session,
            role='assistant',
            content='I run Friday book club with Alex and Mia.',
            character=self.character,
        )

    def test_task_skips_when_user_disables_long_term_memory(self):
        from chat.models import UserProfile
        from chat.tasks import sync_long_term_memory
        UserProfile.objects.update_or_create(
            user=self.user,
            defaults={'allow_long_term_memory': False},
        )

        result = sync_long_term_memory(
            message_id=self.assistant_message.id,
            chat_session_id=self.session.id,
            character_id=self.character.id,
        )

        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(result['reason'], 'user_disabled_long_term_memory')

    def test_task_skips_when_session_private_mode_active(self):
        from chat.models import UserProfile
        from chat.tasks import sync_long_term_memory
        UserProfile.objects.update_or_create(
            user=self.user,
            defaults={'allow_long_term_memory': True},
        )
        self.session.is_private_mode = True
        self.session.save(update_fields=['is_private_mode', 'updated_at'])

        result = sync_long_term_memory(
            message_id=self.assistant_message.id,
            chat_session_id=self.session.id,
            character_id=self.character.id,
        )

        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(result['reason'], 'private_mode')

    def test_task_persists_create_memory_tool_call_with_audit_log(self):
        """Happy-path E2E: a canned OpenAI tool-call response flows through
        ``_collect_memory_actions`` → ``_execute_memory_crud_tool`` →
        ``MemoryManager.create_item`` and the resulting audit-log row is
        written with ``source=CELERY_WORKER``.
        """
        from chat.models import MemoryAuditAction, MemoryAuditLog, MemoryAuditSource, UserProfile
        from chat.tasks import sync_long_term_memory

        UserProfile.objects.update_or_create(
            user=self.user,
            defaults={'allow_long_term_memory': True},
        )

        canned_tool_call_response = {
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': '',
                    'tool_calls': [{
                        'id': 'call_memory_1',
                        'type': 'function',
                        'function': {
                            'name': 'create_memory',
                            'arguments': json.dumps({
                                'section': 'identity',
                                'description': 'Prefers warm drinks.',
                                'reason': 'User mentioned their morning ritual in chat.',
                            }),
                        },
                    }],
                },
            }],
        }
        terminal_text_response = {
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': 'noop',
                    'tool_calls': [],
                },
            }],
        }

        with patch('chat.tasks._request_openai_compatible_completion') as mock_request:
            mock_request.side_effect = [canned_tool_call_response, terminal_text_response]
            with patch('chat.tasks._get_runtime_model_config') as runtime_config:
                runtime_config.return_value = {
                    'provider': 'openai_compatible',
                    'model_name': 'gpt-4.1-mini',
                    'api_key': 'dummy',
                    'base_url': 'https://example.com/v1',
                }
                result = sync_long_term_memory(
                    message_id=self.assistant_message.id,
                    chat_session_id=self.session.id,
                    character_id=self.character.id,
                )

        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['actions'], 1)

        items = list(self.character.memory_items.all())
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].section, 'identity')
        self.assertEqual(items[0].description, 'Prefers warm drinks.')

        rows = list(MemoryAuditLog.objects.filter(character=self.character).order_by('created_at', 'id'))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].action, MemoryAuditAction.CREATE)
        self.assertEqual(rows[0].source, MemoryAuditSource.CELERY_WORKER)
        self.assertEqual(rows[0].after_description, 'Prefers warm drinks.')
